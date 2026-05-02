# analysis_engine.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from scipy.stats import shapiro, lognorm
from scipy.linalg import cholesky, LinAlgError
import torch
import gpytorch
import config
import data_manager
import logging
logger = logging.getLogger(__name__)


def _calculate_patient_data_metrics(patient_data: pd.DataFrame) -> dict:
    """
    【V5.0 新增】
    从病人数据中计算基础指标，用于动态调整超参数。
    """
    metrics = {}
    
    # 1. 计算中位检测间隔
    dates = patient_data.index.to_series().sort_values().drop_duplicates()
    if len(dates) < 3:
        # 数据太少，使用默认值
        metrics['median_interval_days'] = 30.0 
    else:
        # 计算所有日期间隔的中位数
        median_interval = dates.diff().dt.days.median()
        if pd.isna(median_interval) or median_interval < 7:
            # 容错：如果中位数小于7天（不太可能）或计算失败，设为默认值
            metrics['median_interval_days'] = 30.0 
        else:
            metrics['median_interval_days'] = median_interval
            
    # 2. 计算总样本量
    metrics['total_samples'] = len(patient_data)
    
    logger.info(f"[V5.0] 动态参数基础指标: {metrics}")
    return metrics

def _get_dynamic_hyperparameters(
    patient_data: pd.DataFrame, 
    feature_columns: list, 
    initial_weights: pd.Series, 
    shap_stats_df: pd.DataFrame
) -> dict:
    """
    【V5.0 新增】
    动态计算所有调权超参数，消除“魔法数字”。
    """
    params = {}
    metrics = _calculate_patient_data_metrics(patient_data)
    
    # --- 1. 动态计算 SHAP 衰减半衰期 ---
    # 策略：半衰期 = 3倍的中位检测间隔。
    # (例如：如果21天复查一次，半衰期约为63天，即3次复查前的经验减半)
    params['SHAP_DECAY_HALF_LIFE_DAYS'] = metrics['median_interval_days'] * 3.0
    
    # --- 2. 动态计算 Top N 指标 ---
    # 策略：总特征数的 20%，但最少 3 个，最多 8 个。
    total_features = len(feature_columns)
    params['TOP_N_SENSITIVE_INDICATORS'] = max(3, min(8, int(total_features * 0.2)))
    
    # --- 3. 动态计算“滥竽充数”阈值 ---
    # 策略：初始权重必须在 "前 25%" (quantile 0.75)
    params['FREELOADER_THRESHOLD_HIGH'] = max(1.5, initial_weights.quantile(0.75))
    # 策略：SHAP 贡献必须在 "后 25%" (quantile 0.25)
    if not shap_stats_df.empty and (shap_stats_df['shap_impact'] > 0).any():
        # 只在有正贡献的特征里取分位数，更鲁棒
        positive_impacts = shap_stats_df['shap_impact'][shap_stats_df['shap_impact'] > 0]
        params['FREELOADER_THRESHOLD_LOW'] = positive_impacts.quantile(0.25)
    else:
        params['FREELOADER_THRESHOLD_LOW'] = 0.01 # 回退到固定值
        
    # --- 4. 动态计算“奖励因子” (学习率) ---
    # 策略：系统对经验的“信心”随样本量增加而增加。
    # 样本量为 30 时，因子为 1.0。
    # 样本量为 15 时，因子为 0.5 (学习更保守)。
    # 样本量为 60 时，因子为 2.0 (学习更激进)。
    # (设置 0.5 和 2.5 的上下限)
    learning_rate = max(0.5, min(2.5, metrics['total_samples'] / 30.0))
    params['CONSISTENCY_REWARD_FACTOR'] = 0.5 * learning_rate # 一致性奖励基础为 0.5 * 学习率
    params['IMPACT_REWARD_FACTOR'] = 1.0 * learning_rate      # 影响力奖励基础为 1.0 * 学习率

    # --- 5. 保留“策略”和“安全”常量 ---
    # 这些是规则，不是推导值，保留它们是正确的。
    params['FREELOADER_PENALTY'] = 0.5      # 策略：惩罚力度
    params['FINAL_WEIGHT_CLIP_MIN'] = 0.5   # 安全：最小权重
    params['FINAL_WEIGHT_CLIP_MAX'] = 4.0   # 安全：最大权重

    logger.info(f"[V5.0] 动态计算超参数: {params}")
    return params


def detect_trend_change(values: np.ndarray, context: dict = None) -> dict:
    """
    【V5.3 修正】检测时间序列中的突变/急剧变化（考虑临床上下文）
    
    :param values: 时间序列数据
    :param context: 临床上下文字典，包含：
        - 'phase': '稳定监控期' or '强效治疗期'
        - 'baseline_value': float (首次异常时的值)
        - 'upper_limit': float (正常上限)
    :return: 突变检测结果字典
    """
    if len(values) < 3:
        return {
            'has_spike': False,
            'spike_ratio': 1.0,
            'recent_trend': 'stable',
            'threshold': 0.5,
            'context_adjusted': False
        }
    
    # 计算近期和历史的均值
    recent_mean = values[-3:].mean()
    historical_mean = values[:-3].mean() if len(values) > 3 else values.mean()
    
    # 计算变化率
    if abs(historical_mean) < 1e-6:
        spike_ratio = 1.0
    else:
        spike_ratio = abs(recent_mean - historical_mean) / abs(historical_mean)
    
    # 计算动态基础阈值
    if len(values) > 3:
        historical_std = values[:-3].std()
        historical_cv = historical_std / (abs(historical_mean) + 1e-6)
        base_threshold = max(0.3, min(2.0 * historical_cv, 0.8))
    else:
        base_threshold = 0.5
    
    # 根据临床上下文调整阈值
    context_adjusted = False
    if context:
        phase = context.get('phase', '稳定监控期')
        upper_limit = context.get('upper_limit')
        baseline_value = context.get('baseline_value')
        
        # 场景1：稳定期 + 已超标 → 降低阈值（更敏感）
        if phase == '稳定监控期' and upper_limit and historical_mean > upper_limit:
            base_threshold *= 0.5
            context_adjusted = True
            logger.debug(f"  突变检测：稳定期超标，阈值降低至 {base_threshold:.2f}")
            
        # 场景2：治疗期 + 下降趋势 → 提高阈值（更宽容）
        elif phase == '强效治疗期' and recent_mean < historical_mean:
            base_threshold *= 1.5
            context_adjusted = True
            logger.debug(f"  突变检测：治疗期下降，阈值提高至 {base_threshold:.2f}")
            
        # 场景3：低基线的快速上升 → 降低阈值
        if baseline_value and historical_mean < baseline_value * 2:
            base_threshold *= 0.7
            context_adjusted = True
            logger.debug(f"  突变检测：低基线快速上升，阈值降低至 {base_threshold:.2f}")
    
    # 判断是否有突变
    if spike_ratio > base_threshold:
        has_spike = True
        recent_trend = 'sharp_increase' if recent_mean > historical_mean else 'sharp_decrease'
    else:
        has_spike = False
        recent_trend = 'stable'
    
    return {
        'has_spike': has_spike,
        'spike_ratio': spike_ratio,
        'recent_trend': recent_trend,
        'threshold': base_threshold,
        'context_adjusted': context_adjusted
    }



def get_adaptive_lengthscale_prior(df_subset: pd.DataFrame, indicator_name: str = None) -> tuple:
    """
    【V5.2 新增】
    根据数据特征动态调整 lengthscale 先验
    
    :param df_subset: 训练数据子集（包含日期索引）
    :param indicator_name: 指标名称（可选，用于针对性调整）
    :return: (prior_mean, prior_std, constraint_lower, constraint_upper)
    """
    # 计算平均采样间隔 (使用中位数更稳健)
    intervals = df_subset.index.to_series().diff().dt.days.dropna()
    median_interval = max(1.0, intervals.median()) # 避免间隔小于1天

    # 计算数据时间跨度
    time_span = max(1.0, (df_subset.index.max() - df_subset.index.min()).days) # 避免跨度为0

    # 根据指标类型调整因子
    if indicator_name:
        tumor_markers = ['癌胚抗原 CEA', '糖类抗原 19-9', '甲胎蛋白 AFP', '糖类抗原 125']
        slow_vars = ['白蛋白 ALB', '总蛋白 TP', '前白蛋白 PA']
        
        if indicator_name in tumor_markers:
            lengthscale_factor = 2.0  # 肿瘤标志物变化快，使用较短的相关长度
        elif indicator_name in slow_vars:
            lengthscale_factor = 5.0  # 营养指标变化慢，使用较长的相关长度
        else:
            lengthscale_factor = 3.0  # 默认
    else:
        lengthscale_factor = 3.0

    # 策略：先验均值 = max(因子*中位间隔, 时间跨度/3)
    # 确保先验至少覆盖几次采样，同时考虑总时长
    prior_mean = max(median_interval * lengthscale_factor, time_span / 3.0)
    prior_mean = max(prior_mean, 14.0) # 保证最小均值（例如2周）

    # 标准差 = 均值的 50%（允许充分探索）
    prior_std = prior_mean * 0.5

    # 约束：下限 = 中位采样间隔，上限 = 数据总跨度的1.5倍
    # 确保 lengthscale 不会小于采样频率，也不会远超数据范围
    constraint_lower = max(median_interval, 7.0) # 下限至少7天
    constraint_upper = time_span * 1.5
    constraint_upper = max(constraint_upper, constraint_lower + 1.0) # 确保上界大于下界

    logger.info(
        f"  MOGP Prior: 动态 Lengthscale 先验: "
        f"Mean={prior_mean:.1f}天, Std={prior_std:.1f}天, "
        f"Constraint=[{constraint_lower:.1f}, {constraint_upper:.1f}]"
    )

    # 返回先验参数和约束边界
    return prior_mean, prior_std, constraint_lower, constraint_upper



class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, df_subset, indicator_names=None, use_spectral=False):
        """
        增强版多任务GP模型
        
        参数:
            use_spectral: 是否使用SpectralMixture核（更鲁棒，适合复杂模式）
        """
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        # 条件选择核函数
        if use_spectral:
            # SpectralMixture核：更适合复杂、非平稳模式
            print("  📊 使用SpectralMixture核（鲁棒性更强）")
            base_kernel = gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=3,  # 3个高斯混合分量
                ard_num_dims=1
            )
        else:
            # 原有的Matern + Linear组合核
            print("  📊 使用Matern+Linear组合核（标准方案）")

            # --- 应用动态贝叶斯先验 ---
            logger.info("  MOGP Kernel: 应用动态贝叶斯先验")
            # 如果是多任务，使用第一个指标名作为代表
            representative_indicator = indicator_names[0] if indicator_names else None
            prior_mean, prior_std, constraint_lower, constraint_upper = get_adaptive_lengthscale_prior(
                df_subset, 
                indicator_name=representative_indicator
            )

            # Matern 核（捕捉短期波动，带有动态先验）
            matern_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    # 使用动态计算的先验和约束
                    lengthscale_prior=gpytorch.priors.NormalPrior(prior_mean, prior_std),
                    lengthscale_constraint=gpytorch.constraints.Interval(constraint_lower, constraint_upper),
                ),
                outputscale_constraint=gpytorch.constraints.Interval(0.1, 10.0)
            )
            
            # Linear Kernel（捕捉长期趋势）
            linear_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel(
                    variance_constraint=gpytorch.constraints.Interval(0.5, 20.0)
                ),
                outputscale_constraint=gpytorch.constraints.Interval(1.0, 10.0),
                outputscale_prior=gpytorch.priors.NormalPrior(3.0, 1.5)
            )
            
            # 组合核
            base_kernel = matern_kernel + linear_kernel
        
        # 动态调整rank
        task_rank = min(num_tasks, 2) if num_tasks > 1 else 1
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, 
            num_tasks=num_tasks, 
            rank=task_rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    计算两组独立数据之间的 Cohen's d 效应量。
    Cohen's d 是衡量两组均值差异大小的标准化指标。
    """
    n1, n2 = len(group1), len(group2)
    # 如果样本量不足以计算方差，则效应量为0
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    # 计算合并标准差 (pooled standard deviation)
    pooled_std_denominator = n1 + n2 - 2
    if pooled_std_denominator <= 0:
        logger.debug("cohen_d: 合并标准差分母为0，返回 0.0。")
        return 0.0
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / pooled_std_denominator)
    if pooled_std == 0:
        logger.debug("cohen_d: 合并标准差为0，返回 0.0。")
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return abs(d)


def leave_one_out_cross_validation(df_subset: pd.DataFrame, target_indicators: list, 
                                   scalers: dict, num_tasks: int) -> dict:
    """
    留一法交叉验证 (Leave-One-Out Cross-Validation)
    
    用于评估GP模型的预测精度和泛化能力
    
    返回:
        cv_results: 包含每个指标的RMSE、MAE等指标
    """
    logger.info("\n" + "="*60)
    logger.info("🔬 开始留一法交叉验证 (LOO-CV)")
    logger.info("="*60)
    
    cv_errors = {indicator: [] for indicator in target_indicators}
    
    for held_out_idx in range(len(df_subset)):
        logger.debug(f"  LOO-CV: 正在验证第 {held_out_idx+1}/{len(df_subset)} 个点...")
        # 创建训练集（排除当前点）
        train_mask = np.ones(len(df_subset), dtype=bool)
        train_mask[held_out_idx] = False
        
        train_df = df_subset[train_mask]
        test_date = df_subset.index[held_out_idx]
        test_values = df_subset.iloc[held_out_idx]
        
        # 检查训练集大小
        if len(train_df) < 3:
            logger.warning(f"  LOO-CV: ⚠️ 跳过索引 {held_out_idx}：训练数据不足 ({len(train_df)} < 3)")
            continue
        
        # 准备训练数据
        first_day = train_df.index.min()
        train_x = torch.tensor((train_df.index - first_day).days.values, dtype=torch.float64)
        
        # 标准化
        train_y_list = []
        for indicator in target_indicators:
            values = train_df[indicator].values.reshape(-1, 1)
            values_scaled = scalers[indicator].transform(values).flatten()
            train_y_list.append(values_scaled)

        train_y = torch.tensor(np.column_stack(train_y_list), dtype=torch.float64)

        if train_x.ndim == 1:
            train_x = train_x.unsqueeze(-1)
        
        # 快速训练（减少迭代次数）
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            noise_constraint=gpytorch.constraints.Interval(1e-3, 1.0)
        )
        model = MultitaskGPModel(
            train_x, 
            train_y, 
            likelihood, 
            num_tasks, 
            df_subset=train_df, 
            use_spectral=False
        )
        model = model.double()
        likelihood = likelihood.double()
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # 简化训练（只训练50轮）
        for _ in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        # 预测
        model.eval()
        likelihood.eval()

        test_x = torch.tensor([(test_date - first_day).days], dtype=torch.float64).unsqueeze(-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(test_x))
            pred_mean_scaled = pred.mean[0].numpy()
        
        # 逆标准化并计算误差
        for i, indicator in enumerate(target_indicators):
            pred_value = scalers[indicator].inverse_transform(
                pred_mean_scaled[i].reshape(-1, 1)
            )[0, 0]
            true_value = test_values[indicator]
            error = abs(pred_value - true_value)
            cv_errors[indicator].append(error)
    
    # 计算统计指标
    cv_results = {}
    logger.info("\n📊 LOO-CV 交叉验证结果:")
    logger.info("-" * 60)
    for indicator in target_indicators:
        errors = np.array(cv_errors[indicator])
        if len(errors) > 0:
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(errors)
            max_error = np.max(errors)
            
            cv_results[indicator] = {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'n_folds': len(errors)
            }
            
            logger.info(f"  {indicator}:")
            logger.info(f"    - RMSE: {rmse:.3f}")
            logger.info(f"    - MAE: {mae:.3f}")
            logger.info(f"    - 最大误差: {max_error:.3f}")
            logger.info(f"    - 折数: {len(errors)}")
        else:
            logger.warning(f"  {indicator}: 无有效LOO-CV验证数据")
            cv_results[indicator] = None

    logger.info("="*60 + "\n")
    return cv_results



def residual_diagnostics(true_values: np.ndarray, predicted_values: np.ndarray, 
                        indicator_name: str) -> dict:
    """
    残差诊断函数
    
    检查模型假设是否成立：
    1. 正态性检验 (Shapiro-Wilk test)
    2. 自相关性检验
    3. 异方差性检验
    
    返回:
        diagnostics: 诊断结果字典
    """
    logger.debug(f"执行残差诊断: {indicator_name}, {len(true_values)} 个数据点。")
    residuals = true_values - predicted_values
    
    # 1. 正态性检验
    if len(residuals) >= 3:
        stat, p_value = shapiro(residuals)
        is_normal = p_value > 0.05
    else:
        stat, p_value = None, None
        is_normal = None
    
    # 2. 自相关性（简化版：相邻残差的相关系数）
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = None
    
    # 3. 异方差性（残差的标准差是否随预测值变化）
    residual_std = np.std(residuals)
    mean_residual = np.mean(np.abs(residuals))
    
    diagnostics = {
        'mean_residual': mean_residual,
        'std_residual': residual_std,
        'shapiro_stat': stat,
        'shapiro_pvalue': p_value,
        'is_normal': is_normal,
        'autocorr': autocorr
    }
    
    logger.info(f"\n🔬 残差诊断 - {indicator_name}:")
    logger.info("-" * 50)
    logger.info(f"  平均残差: {mean_residual:.3f}")
    logger.info(f"  残差标准差: {residual_std:.3f}")
    if p_value is not None:
        logger.info(f"  Shapiro-Wilk检验: p={p_value:.4f} {'✅ 正态' if is_normal else '❌ 非正态'}")
    if autocorr is not None:
        logger.info(f"  自相关系数: {autocorr:.3f} {'⚠️ 强自相关' if abs(autocorr) > 0.5 else '✅ 无显著自相关'}")

    return diagnostics



def calculate_sensitivity_weights(df: pd.DataFrame, feature_columns: list, 
                                  context_columns_to_exclude: list) -> tuple:
    """
    【V5.2 修正】允许单阶段训练
    
    返回:
        - weights: (pd.Series) 权重
        - fallback_activated: (bool) 是否启动回退
        - single_phase_mode: (bool) 【新增】是否为单阶段模式
    """
    logger.info("开始计算指标敏感度权重 (Cohen's d 冷启动)...")

    # 二次过滤
    valid_feature_columns = [
        col for col in feature_columns 
        if col not in context_columns_to_exclude
        and '_transition_shock' not in col
    ]
    
    filtered_count = len(feature_columns) - len(valid_feature_columns)
    if filtered_count > 0:
        logger.debug(
            f"  ℹ️ 已过滤 {filtered_count} 个不参与权重计算的特征\n"
            f"    （包括上下文列和过渡特征）"
        )

    treatment_phases = ['强效治疗期']
    monitoring_phases = ['稳定监控期']
    
    treatment_data = df[df['phase'].isin(treatment_phases)]
    monitoring_data = df[df['phase'].isin(monitoring_phases)]
    
    # 检查是否至少有一个阶段的数据
    if treatment_data.empty and monitoring_data.empty:
        logger.warning("无法计算敏感度权重：'强效治疗期' 和 '稳定监控期' 数据均为空。")
        logger.warning("将激活回退模式（所有权重为1.0）。")
        return None, True, False
    
    # 处理单阶段情况
    if treatment_data.empty:
        logger.warning("⚠️ 缺少'强效治疗期'数据，将使用单阶段模式（基于稳定期内部变异性）")
        return _calculate_single_phase_weights(monitoring_data, feature_columns, context_columns_to_exclude), False, True
    
    if monitoring_data.empty:
        logger.warning("⚠️ 缺少'稳定监控期'数据，将使用单阶段模式（基于治疗期内部变异性）")
        return _calculate_single_phase_weights(treatment_data, feature_columns, context_columns_to_exclude), False, True
    
    # 正常双阶段计算
    logger.info("检测到双阶段数据，进行 Cohen's d 计算...")
    valid_feature_columns = [
        col for col in feature_columns 
        if col not in context_columns_to_exclude
    ]
    
    sensitivity_scores = pd.Series(index=valid_feature_columns, dtype=float)
    for col in valid_feature_columns:
        group1 = treatment_data[col].dropna()
        group2 = monitoring_data[col].dropna()
        if len(group1) >= 2 and len(group2) >= 2:
            sensitivity_scores[col] = cohen_d(group1, group2)

    sensitivity_scores = sensitivity_scores.fillna(0.0)

    min_d, max_d = 0, sensitivity_scores.max()
    min_weight, max_weight = 1.0, 3.0

    if max_d > 0:
        weights = min_weight + (sensitivity_scores - min_d) * (max_weight - min_weight) / (max_d - min_d)
    else:
        logger.warning("所有指标的 Cohen's d 均为0，使用默认权重 1.0。")
        weights = pd.Series(1.0, index=valid_feature_columns)
        
    final_weights = pd.Series(1.0, index=feature_columns)
    final_weights.update(weights)
        
    logger.info(f"敏感度权重计算完成。均值: {final_weights.mean():.2f}, 范围: [{final_weights.min():.2f}, {final_weights.max():.2f}]")
    
    return final_weights, False, False  # (权重, 未回退, 非单阶段)


def _calculate_single_phase_weights(phase_data: pd.DataFrame, 
                                    feature_columns: list, 
                                    context_columns: list) -> pd.Series:
    """
    【V5.2 新增】基于单阶段数据内部变异性计算权重
    
    策略：变异系数（CV）越大 → 权重越高（认为该指标在该阶段更"敏感"）
    """
    logger.info("使用单阶段模式计算权重（基于内部变异性）...")
    
    valid_features = [f for f in feature_columns if f not in context_columns]
    cv_scores = pd.Series(index=valid_features, dtype=float)
    
    for col in valid_features:
        values = phase_data[col].dropna()
        if len(values) >= 3:  # 至少3个点才能计算有意义的 CV
            mean_val = values.mean()
            std_val = values.std()
            if mean_val > 1e-6:
                cv_scores[col] = std_val / mean_val
            else:
                cv_scores[col] = 0.0
        else:
            cv_scores[col] = 0.0
    
    cv_scores = cv_scores.fillna(0.0)
    
    # 将 CV 映射到权重范围 [1.0, 2.0]（比 Cohen's d 的范围小，因为信息量更少）
    min_cv, max_cv = 0, cv_scores.max()
    min_weight, max_weight = 1.0, 2.0
    
    if max_cv > 0:
        weights = min_weight + (cv_scores - min_cv) * (max_weight - min_weight) / (max_cv - min_cv)
    else:
        weights = pd.Series(1.0, index=valid_features)
    
    final_weights = pd.Series(1.0, index=feature_columns)
    final_weights.update(weights)
    
    logger.info(f"单阶段权重计算完成。均值: {final_weights.mean():.2f}")
    return final_weights



def discover_dynamic_indicators(
    patient_data: pd.DataFrame, 
    feature_columns: list, 
    context_columns: list, 
    patient_id: int
) -> tuple:
    """
    【双锚点混合版 + 归因类型加权 + 时间衰减】
    融合"客观统计"和"主观反馈"来动态修正特征权重。
    
    核心创新：
    1. W_Objective（客观锚点）：基于 Cohen's d 统计量
    2. W_Subjective（主观锚点）：基于用户反馈的 HITL 经验
    3. Alpha（信任因子）：根据反馈数量动态调整两者权重
    4. SHAP 归因类型加权：真实归因 100% vs 代理归因 30%
    5. 安全护栏：clip 操作防止极端值
    
    优势：
    - 避免 SHAP 的循环论证陷阱
    - 充分利用用户反馈（但不盲目信任）
    - 冷启动时依赖客观统计，成熟后融合主观经验
    
    返回:
        - final_weights: 混合后的最终权重
        - boost_df: 调权详情（含客观/主观/混合三列）
        - dynamic_params: 动态参数
        - fallback_activated: 是否回退
    """
    logger.info("开始发现动态指标  (V8.1 - 反馈时间衰减版)...")

    # --- 时间衰减参数 ---
    FEEDBACK_HALF_LIFE_DAYS = 365.0
    CURRENT_DATE = pd.Timestamp.now()
    logger.info(f"  > 应用反馈时间衰减 (半衰期: {FEEDBACK_HALF_LIFE_DAYS}天, 基准日期: {CURRENT_DATE.date()})")    
    
    # 0: 性能优化 - 转为集合
    feature_columns_set = set(feature_columns)
    
    # 1: 计算客观锚点
    logger.info("1. 计算客观锚点（Cohen's d）...")
    w_objective, fallback_activated, single_phase_mode = calculate_sensitivity_weights(
        patient_data, feature_columns, context_columns_to_exclude=context_columns
    )
    
    if fallback_activated or w_objective is None:
        logger.warning("Cohen's d 权重计算失败，激活回退模式。")
        default_weights = pd.Series(1.0, index=feature_columns)
        return default_weights, pd.DataFrame(), {}, True
    
    logger.info(f"  ✅ 客观权重范围: [{w_objective.min():.2f}, {w_objective.max():.2f}]")
    
    # 2: 加载用户反馈
    logger.info("\n2. 加载用户反馈数据...")
    feedback_data = data_manager.load_all_feedback_with_shap(patient_id)
    
    if not feedback_data:
        logger.info("  ⚠️ 无用户反馈数据，跳过主观锚点计算")
        boost_df = pd.DataFrame(index=feature_columns)
        boost_df['W_Objective (Cohen\'s d)'] = w_objective
        boost_df['W_Subjective (HITL)'] = 1.0
        boost_df['Alpha (信任因子)'] = 0.0
        boost_df['W_Final'] = w_objective
        
        params = _get_dynamic_hyperparameters(
            patient_data, feature_columns, w_objective, pd.DataFrame()
        )
        return w_objective, boost_df, params, False
    
    logger.info(f"  ✅ 加载了 {len(feedback_data)} 条反馈记录")

    # 统计归因类型分布
    real_count = sum(1 for r in feedback_data if r.get('shap_type') == 'real')
    proxy_count = sum(1 for r in feedback_data if r.get('shap_type') == 'proxy')
    logger.info(
        f"  📊 归因类型分布: Real={real_count} ({real_count/len(feedback_data)*100:.1f}%), "
        f"Proxy={proxy_count} ({proxy_count/len(feedback_data)*100:.1f}%)"
    )

    # 3: 计算主观锚点
    feature_confirmations = pd.Series(0.0, index=feature_columns)
    feature_rejections = pd.Series(0.0, index=feature_columns)
    feature_feedback_counts = pd.Series(0, index=feature_columns)
    
    # 定义基础学习率映射
    BASE_LR_MAP = {
        'model_anomaly': 1.0,        # IsolationForest 警报（多特征，最可靠）
        'baseline_deviation': 0.7,   # Z-Score 基线偏离（统计方法，较可靠）
        'historical_similarity': 0.8, # DTW 历史匹配（经验方法，较可靠）
        'heuristic': 0.5,            # 单指标超标等规则（主观性强）
        'unknown': 0.6               # 未知类型（回退值）
    }
    
    logger.info(f"\n  加载了 {len(feedback_data)} 条反馈记录，开始解析...")
    
    for record in feedback_data:
        label = record['label']
        shap_values = record['shap_values']
        pattern_type = record.get('pattern_type', 'unknown')
        shap_type = record.get('shap_type', 'proxy')
        
        # --- 时间衰减计算（带防御性检查）---
        feedback_timestamp = pd.Timestamp(record.get('timestamp'))
        
        if feedback_timestamp is None or pd.isna(feedback_timestamp):
            logger.warning(
                f"  反馈 {record.get('observation_uuid', 'N/A')[:8]}... 缺少时间戳，"
                f"使用保守衰减（假设1年前）"
            )
            time_decay_weight = 0.5
            days_ago_display = "365 (默认)"
        else:
            days_ago_raw = (CURRENT_DATE - feedback_timestamp).days
            
            if days_ago_raw < 0:
                logger.warning(
                    f"  ⚠️ 异常时间戳（未来日期）：{feedback_timestamp}，"
                    f"已修正为0天"
                )
                days_ago = 0
            else:
                days_ago = days_ago_raw
            
            time_decay_weight = 0.5 ** (days_ago / FEEDBACK_HALF_LIFE_DAYS)
            days_ago_display = str(days_ago)
        
        # --- 计算基础学习率（已包含时间衰减）---
        base_lr = BASE_LR_MAP.get(pattern_type, 0.6)
        base_lr_with_time = base_lr * time_decay_weight
        
        # --- SHAP 置信度 ---
        shap_confidence = 1.0 if shap_type == 'real' else 0.3
        
        logger.debug(
            f"\n  📋 反馈: {record['observation_uuid'][:8]}... "
            f"({label}, {pattern_type}, SHAP={shap_type})\n"
            f"    - Base LR: {base_lr:.2f}\n"
            f"    - 时间衰减: {time_decay_weight:.2f} ({days_ago_display}天前)\n"
            f"    - 衰减后 LR: {base_lr_with_time:.2f}\n"
            f"    - SHAP Confidence: {shap_confidence:.2f}"
        )
        
        # 方向
        if label == 'significant':
            direction = +1.0
        elif label == 'benign':
            direction = -0.5
        elif label == 'lab_error':
            direction = -1.0
        else:
            continue
        
        # 先过滤有效特征
        valid_shap_features = {
            k: v for k, v in shap_values.items() 
            if k in feature_columns_set
        }
        
        if not valid_shap_features:
            logger.debug(f"  ⚠️ 反馈不包含有效特征，跳过")
            continue
        
        # 按 SHAP 分配权重
        total_abs_shap = sum(abs(v) for v in valid_shap_features.values())
        
        if total_abs_shap < 1e-6:
            # 单特征归因（通常是代理 SHAP）
            weight_per_feature = 1.0 / len(valid_shap_features)
            
            for feature in valid_shap_features.keys():
                feedback_count = feature_feedback_counts.get(feature, 0)
                trust_multiplier = 0.5 + 0.5 * np.tanh(feedback_count / 5.0)
                
                # 使用已包含时间衰减的 base_lr_with_time
                final_lr = base_lr_with_time * shap_confidence * trust_multiplier * weight_per_feature * direction
                
                if direction > 0:
                    feature_confirmations[feature] += final_lr
                else:
                    feature_rejections[feature] += abs(final_lr)
                
                logger.debug(
                    f"    - {feature}: \n"
                    f"        归因: 单特征, 反馈次数={feedback_count}, 信任度={trust_multiplier:.2f}\n"
                    f"        最终学习率={final_lr:+.3f} "
                    f"(当前累积: 确认={feature_confirmations[feature]:.2f}, "
                    f"否认={feature_rejections[feature]:.2f})"
                )
                            
                # 在分配完后才增加计数
                feature_feedback_counts[feature] += 1
        else:
            # 多特征归因（通常是真实 SHAP）
            for feature, shap_val in valid_shap_features.items():
                # 先读取当前计数
                feedback_count = feature_feedback_counts.get(feature, 0)
                trust_multiplier = 0.5 + 0.5 * np.tanh(feedback_count / 5.0)
                
                normalized_contribution = abs(shap_val) / total_abs_shap
                
                final_lr = base_lr_with_time * shap_confidence * trust_multiplier * normalized_contribution * direction
                
                if direction > 0:
                    feature_confirmations[feature] += final_lr
                else:
                    feature_rejections[feature] += abs(final_lr)
                
                logger.debug(
                    f"    - {feature}: \n"
                    f"        归因: SHAP={shap_val:.2f} ({normalized_contribution:.1%}), "
                    f"反馈次数={feedback_count}, 信任度={trust_multiplier:.2f}\n"
                    f"        最终学习率={final_lr:+.3f} "
                    f"(当前累积: 确认={feature_confirmations[feature]:.2f}, "
                    f"否认={feature_rejections[feature]:.2f})"
                )
                
                # 在分配完后才增加计数
                feature_feedback_counts[feature] += 1
    
    # 3.3 计算主观权重
    w_subjective = pd.Series(1.0, index=feature_columns)
    
    logger.info(f"\n  📊 主观权重计算详情（Top 10 受影响特征）:")
    
    weight_changes = pd.DataFrame({
        'confirmations': feature_confirmations,
        'rejections': feature_rejections,
        'feedback_count': feature_feedback_counts
    })
    weight_changes['impact'] = (weight_changes['confirmations'] - weight_changes['rejections']).abs()
    top_impacted = weight_changes.nlargest(10, 'impact')
    
    for feature in top_impacted.index:
        confirmations = feature_confirmations[feature]
        rejections = feature_rejections[feature]
        trust_score = 1.0 + (confirmations * 0.2) - (rejections * 0.2)
        w_subjective[feature] = np.clip(trust_score, 0.5, 2.0)
        
        logger.info(
            f"    - {feature}:\n"
            f"        确认分: {confirmations:+.2f}, "
            f"否认分: {rejections:.2f}, "
            f"反馈次数: {feature_feedback_counts[feature]}\n"
            f"        原始分数: {trust_score:.2f}, "
            f"Clip后: {w_subjective[feature]:.2f}"
        )
    
    # 处理剩余特征（无日志）
    for feature in feature_columns:
        if feature in top_impacted.index:
            continue
        confirmations = feature_confirmations[feature]
        rejections = feature_rejections[feature]
        trust_score = 1.0 + (confirmations * 0.2) - (rejections * 0.2)
        w_subjective[feature] = np.clip(trust_score, 0.5, 2.0)
    
    logger.info(f"\n  ✅ 主观权重范围: [{w_subjective.min():.2f}, {w_subjective.max():.2f}]")
    
    # 4: 计算信任因子 Alpha
    logger.info("\n4. 计算信任因子 Alpha...")
    total_feedback = len(feedback_data)
    alpha = 0.5 * (1 - np.exp(-total_feedback / 20))  # 上限从 0.3 → 0.5
    
    logger.info(
        f"  ✅ Alpha = {alpha:.3f} "
        f"(反馈数={total_feedback}, 主观占比={alpha*100:.1f}%)"
    )
    
    # 5: 混合锚点
    logger.info("\n5. 混合客观与主观锚点...")
    w_final = (1 - alpha) * w_objective + alpha * w_subjective
    logger.info(f"  ✅ 混合权重范围: [{w_final.min():.2f}, {w_final.max():.2f}]")
    
    # 6: 安全护栏
    logger.info("\n6. 应用安全护栏...")
    params = _get_dynamic_hyperparameters(
        patient_data, feature_columns, w_objective, pd.DataFrame()
    )
    
    w_final_clipped = w_final.clip(
        params['FINAL_WEIGHT_CLIP_MIN'],
        params['FINAL_WEIGHT_CLIP_MAX']
    )
    
    clipped_features = w_final[w_final != w_final_clipped]
    if not clipped_features.empty:
        logger.info(f"  ⚠️ {len(clipped_features)} 个特征触及边界约束:")
        for feat in clipped_features.index[:3]:
            logger.info(f"    - {feat}: {w_final[feat]:.2f} → {w_final_clipped[feat]:.2f}")
    else:
        logger.info(f"  ✅ 所有权重在安全范围内")
    
    w_final = w_final_clipped
    
    # 7: 构建调权详情表（修复除零风险）
    boost_df = pd.DataFrame(index=feature_columns)
    boost_df.index.name = "特征"
    boost_df['W_Objective (Cohen\'s d)'] = w_objective
    boost_df['W_Subjective (HITL)'] = w_subjective
    boost_df['Alpha (信任因子)'] = alpha
    boost_df['W_Final (混合)'] = w_final
    boost_df['调整幅度 (%)'] = (
        (w_final - w_objective) / (w_objective + 1e-8) * 100
    ).round(1)
    
    # 8: 最终日志
    logger.info("\n" + "="*70)
    logger.info("✅ 双锚点混合权重计算完成")
    logger.info("="*70)
    
    top_n = params['TOP_N_SENSITIVE_INDICATORS']
    final_top = w_final.sort_values(ascending=False).head(top_n)
    
    logger.info(f"\n📊 最终 Top {top_n} 指标:")
    for feat in final_top.index:
        logger.info(
            f"  - {feat}: {w_final[feat]:.2f} "
            f"(客观={w_objective[feat]:.2f}, "
            f"主观={w_subjective[feat]:.2f}, "
            f"调整={boost_df.loc[feat, '调整幅度 (%)']:+.1f}%)"
        )
    
    logger.info("="*70 + "\n")
    
    return w_final, boost_df, params, False



def train_baseline_model(df: pd.DataFrame, feature_columns: list, weights: pd.Series) -> dict:
    """
    为“稳定监控期”和“强效治疗期”的数据分别训练独立的健康基线模型。
    
    :return: 一个包含'stable'和/或'treatment'模型的字典。
    """
    logger.info("开始训练健康基线模型 (IsolationForest)...")

    invalid = [c for c in feature_columns if '_transition_shock' in c]
    if invalid:
        logger.warning(
            f"⚠️ 检测到 {len(invalid)} 个污染特征（已自动过滤）：\n"
            f"  {', '.join(invalid[:3])}{'...' if len(invalid) > 3 else ''}\n"
            f"  → 提示：检查 controller.train_models() 的过滤逻辑"
        )
        feature_columns = [c for c in feature_columns if c not in invalid]

    all_models = {}
    
    # 定义我们要建模的两个阶段
    phases_to_model = ['稳定监控期', '强效治疗期'] 
    
    for phase in phases_to_model:
        logger.info(f"--- 正在尝试为 '{phase}' 训练模型 ---")
        phase_data = df[df['phase'] == phase].copy()

        # 检查是否满足最小样本量
        if len(phase_data) >= config.MIN_SAMPLES_FOR_MODEL:
            logger.info(f"'{phase}' 数据 {len(phase_data)} 条，满足最小样本量 {config.MIN_SAMPLES_FOR_MODEL}。")

            current_phase_features = feature_columns.copy()  # 先复制一份

            if phase == '强效治疗期':
                logger.info(f"[智能过滤] '{phase}': 正在过滤 MOGP 专属特征...")
                
                # 识别所有 MOGP 特征列
                mogp_feature_keywords = ['_gp_predicted_mean', '_gp_uncertainty_std', '_gp_uncertainty_trend']
                
                # 过滤掉这些特征
                current_phase_features = [
                    col for col in feature_columns 
                    if not any(keyword in col for keyword in mogp_feature_keywords)
                ]
                
                num_filtered = len(feature_columns) - len(current_phase_features)
                logger.info(
                    f"[智能过滤] '{phase}': 已过滤 {num_filtered} 个 MOGP 特征。"
                    f"特征数量: {len(feature_columns)} -> {len(current_phase_features)}"
                )
            else:
                # 稳定监控期：保留所有特征（包括 MOGP）
                logger.info(f"[智能过滤] '{phase}': 保留所有特征（包括 MOGP）")

            train_data = phase_data[current_phase_features]

            # 计算填充值（使用均值）
            imputation_values = train_data.mean()

            # 第一次填充：使用均值填充
            filled_train_data = train_data.fillna(imputation_values)

            # 第二次填充：对于仍然是 NaN 的列（可能该列全是 NaN），填充为 0
            filled_train_data = filled_train_data.fillna(0)
            logger.debug(f"'{phase}' 训练数据 Shape (填充后): {filled_train_data.shape}")
            aligned_train_data, aligned_weights = filled_train_data.align(
                weights, axis=1, join='inner'
            )

            # 安全性检查
            if aligned_train_data.empty:
                logger.warning(f"'{phase}' 数据与权重对齐后为空，跳过该阶段。")
                continue

            actual_features_trained = aligned_train_data.columns.tolist()
            logger.debug(f"'{phase}' 实际训练特征数量: {len(actual_features_trained)}")
            
            weighted_train_data = aligned_train_data * aligned_weights
            logger.debug(f"'{phase}' 训练数据 Shape (加权后): {weighted_train_data.shape}")
            
            try:
                model = IsolationForest(contamination='auto', random_state=42).fit(weighted_train_data)
                
                # 保存该阶段所需的所有组件
                all_models[phase] = {
                    'model': model, 
                    'imputation': imputation_values,
                    'background_data': weighted_train_data,  # 背景数据就是加权后的训练数据
                    'features_trained': actual_features_trained # 记录实际训练的特征
                }
                logger.info(f"✅ '{phase}' 基线模型 (IsolationForest) 训练成功。")
            except ValueError as e:
                 logger.error(f"❌ '{phase}' 模型训练失败 (可能由于数据全为NaN或常量): {e}", exc_info=True)
                 
        else:
            logger.warning(f"'{phase}' 基线模型训练失败：数据不足 ({len(phase_data)} < {config.MIN_SAMPLES_FOR_MODEL})。")
            
    if not all_models:
        logger.error("所有阶段均未能成功训练模型。")
        return None
        
    return all_models


def get_anomaly_score(model: IsolationForest, weights: pd.Series, new_data_point: pd.DataFrame) -> float:
    """
    使用训练好的模型和权重，计算新数据点的异常分数。
    IsolationForest 的 decision_function 返回值越负，代表越异常。
    """
    logger.debug("计算新数据点的异常分数...")
    weighted_new_point = new_data_point * weights
    score = model.decision_function(weighted_new_point)
    logger.debug(f"计算得到异常分数: {score[0]:.4f}")
    return score[0]


def train_gp_model(df: pd.DataFrame, target_column: str):
    """
    (单任务, sklearn)
    为单一目标指标训练一个高斯过程回归(Gaussian Process Regression)模型。
    :param df: 包含历史数据的DataFrame，索引必须是datetime类型。
    :param target_column: 需要被建模的列名 (例如 '癌胚抗原 CEA')。
    :return: a tuple: (训练好的GP模型, 第一天日期的datetime对象) or (None, None)
    """
    logger.info(f"开始训练单任务GP模型 (sklearn) for '{target_column}'...")
    gp_data = df[[target_column]].dropna()
    if gp_data.shape[0] < 3: # GP模型至少需要几个点来学习趋势
        logger.warning(f"单任务GP训练跳过：'{target_column}' 数据点不足 ({gp_data.shape[0]} < 3)。")
        return None, None
        
    # 将日期索引转换为数值型输入 (距离第一次检测过了多少天)
    first_day = gp_data.index.min()
    X_train = (gp_data.index - first_day).days.values.reshape(-1, 1)
    y_train = gp_data[target_column].values

    # 定义高斯过程的核函数 (Kernel)
    # RBF核用于捕捉平滑的趋势，WhiteKernel用于解释数据中的噪声。
    kernel = 1.0 * RBF(length_scale=90.0, length_scale_bounds=(1e-1, 1e3)) \
             + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))
    
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10, random_state=42)
    
    try:
        gp_model.fit(X_train, y_train)
        logger.info(f"✅ 单任务GP模型 for '{target_column}' 训练成功。")
    except Exception as e:
        logger.error(f"单任务GP模型 for '{target_column}' 训练失败: {e}", exc_info=True)
        return None, None
    
    return gp_model, first_day


def predict_with_gp(gp_model: GaussianProcessRegressor, first_day, target_date):
    """
    (单任务, sklearn)
    使用训练好的GP模型，预测指定日期的指标值、不确定性以及预期的变化趋势（速度）。
    """
    logger.debug(f"GP (sklearn) 预测: {target_date}")
    # 准备要预测的两个时间点：当天 和 第二天
    day_target = (target_date - first_day).days
    day_next = day_target + 1
    X_pred = np.array([day_target, day_next]).reshape(-1, 1)
    
    # 进行预测，返回均值和标准差
    mean_preds, std_preds = gp_model.predict(X_pred, return_std=True)
    
    mean_today = mean_preds[0]
    std_today = std_preds[0]
    
    # 用有限差分法估算瞬时趋势（速度：值/天）
    predicted_trend = mean_preds[1] - mean_preds[0]
    
    return mean_today, std_today, predicted_trend


def train_and_predict_mogp(df: pd.DataFrame, target_indicators: list):
    """
    多任务高斯过程预测 - V5.1 (临床数据适配增强版)
    
    核心改进：
    1. 统一使用对数正态分布计算置信区间下限（增强数值稳定性）
    2. 强制 T-1 和 T=0 置信区间完全对齐
    3. 先计算未来点，再处理历史点（逻辑更清晰）
    4. 保留详细的诊断日志
    5. 动态调整 jitter 根据数据密度
    6. 智能调整 Linear Kernel 学习率
    7. 基于临床采样频率的自适应可靠性评估
    """
    
    # ========== 阶段0: 输入验证 ==========
    if df.empty or not target_indicators:
        logger.error("MOGP: 错误：输入数据为空或未指定目标指标")
        return None

    logger.info("\n" + "="*80)
    logger.info("MOGP训练与预测 - V5.1 (临床数据适配增强版)")
    logger.info("="*80)

    logger.info(f"\n原始数据信息:")
    logger.info(f"  - 数据行数: {len(df)}")
    logger.info(f"  - 目标指标: {target_indicators}")
    logger.info(f"  - 日期范围: {df.index.min()} 至 {df.index.max()}")
    
    df_subset = df[target_indicators].copy()

    # ========== 阶段1: 计算全局采样间隔（用于预测窗口） ==========
    logger.info("\n🎯 MOGP 预测窗口计算（基于数据密度）:")
    intervals = df_subset.index.to_series().diff().dt.days.dropna()
    
    if len(intervals) >= 2:
        global_median_interval = intervals.median()
        global_median_interval = max(1.0, global_median_interval)
        logger.info(f"  ✅ 中位采样间隔: {global_median_interval:.0f}天")
    else:
        # 如果数据点太少，使用保守估计
        global_median_interval = 30.0
        logger.warning(f"  ⚠️ 数据点不足，使用默认采样间隔: {global_median_interval:.0f} 天")
    
    logger.info(f"  💡 此参数将用于：")
    logger.info(f"     - 计算预测窗口长度")
    logger.info(f"     - 生成复查间隔提示")
    logger.info(f"     - 评估MOGP结果的可信度\n")
    
    # ========== 阶段2: 数据预处理（根据指标数量选择策略） ==========
    if len(target_indicators) == 1:
        # 单指标：只删除该指标的缺失值
        indicator = target_indicators[0]
        df_subset = df_subset[df_subset[indicator].notna()]
        logger.info(f"\n  MOGP: 单指标模式：保留 {indicator} 的所有非空值")
    else:
        # 多指标：删除任意列有缺失的行（确保时间点对齐）
        logger.info(f"\n  MOGP: 指标缺失值统计:")
        for col in target_indicators:
            missing_count = df_subset[col].isna().sum()
            logger.info(f"  - {col}: {missing_count}/{len(df_subset)} 个缺失值")

        # 合并同一天的重复数据，取平均值
        df_subset = df_subset.groupby(df_subset.index).mean()
        df_subset = df_subset.dropna()
        logger.info(f"\n  MOGP: 多指标模式：保留完整记录")

    logger.info(f"\n  MOGP: 过滤后的完整数据点:")
    logger.info(f"  - 剩余行数: {len(df_subset)}")
    if len(df_subset) > 0:
        logger.info(f"  - 日期范围: {df_subset.index.min()} 至 {df_subset.index.max()}")

    # ========== 阶段3: 数据量和波动性验证 ==========
    if len(df_subset) < 5:
        logger.error(f"\n  MOGP: 错误：数据点不足({len(df_subset)}个)，至少需要5个稳定期记录")
        return None
    
    # 数据波动性检查
    for indicator in target_indicators:
        values = df_subset[indicator]
        if values.std() < 1e-6:
            logger.error(f"MOGP: 错误：{indicator} 数据无变化(std={values.std():.2e})，无法训练GP")
            return None
    
    # ========== 阶段4: 数据密度分析 ==========
    first_day = df_subset.index.min()
    train_x = torch.tensor((df_subset.index - first_day).days.values, dtype=torch.float64)
    
    time_span_days = (df_subset.index.max() - df_subset.index.min()).days + 1
    data_density = len(train_x) / time_span_days
    logger.info(f"\n数据密度分析:")
    logger.info(f"  - 时间跨度: {time_span_days} 天")
    logger.info(f"  - 数据点数: {len(train_x)} 个")
    logger.info(f"  - 平均密度: {data_density:.4f} 点/天 (平均每{1/data_density:.1f}天1个点)")

    # ========== 阶段5: 标准化和趋势检测 ==========
    scalers = {}
    train_y_scaled_list = []
    trend_info = {}

    logger.info(f"\n应用StandardScaler标准化（保留趋势信息）:")
    for indicator in target_indicators:
        values = df_subset[indicator].values.reshape(-1, 1)

        # 移除极端异常值
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 1e-6:
            z_scores = np.abs((values - mean_val) / std_val)
            outliers = z_scores > 4
            if outliers.any():
                outlier_count = outliers.sum()
                logger.warning(f"  MOGP: ⚠️ {indicator} 检测到 {outlier_count} 个极端异常值 (Z-score > 4)")
                median_val = np.median(values)
                values[outliers] = median_val
                logger.warning(f"     → 已用中位数 {median_val:.2f} 替代")

        # 趋势检测
        trend = detect_trend_change(values.flatten())
        trend_info[indicator] = trend
        
        # 标准化
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values).flatten()
        
        scalers[indicator] = scaler
        train_y_scaled_list.append(values_scaled)
        
        logger.info(f"    - {indicator}:")
        logger.info(f"      原始范围: [{values.min():.2f}, {values.max():.2f}]")
        logger.info(f"      缩放后范围: [{values_scaled.min():.2f}, {values_scaled.max():.2f}]")
        logger.info(f"      均值: {scaler.mean_[0]:.2f}, 标准差: {scaler.scale_[0]:.2f}")
        logger.info(f"      趋势检测: {trend['recent_trend']} (变化率={trend['spike_ratio']:.1%})")

    # 转换为tensor
    train_y = torch.tensor(np.column_stack(train_y_scaled_list), dtype=torch.float64)
    
    # 检测是否有趋势突变
    has_any_spike = any(info['has_spike'] for info in trend_info.values())
    
    if has_any_spike:
        logger.info(f"\n  MOGP: 检测到趋势突变，将通过提高LinearKernel学习率来增强趋势学习...")
        logger.info(f"    数据点数: {len(train_x)} (不进行数据增强)")

    # ========== 阶段6: 计算预测窗口 ==========
    logger.info(f"\n  MOGP: 开始训练MOGP模型（针对数据特征优化）...")

    if train_x.ndim == 1:
        train_x = train_x.unsqueeze(-1)

    num_tasks = len(target_indicators)

    # 策略：预测窗口 = 采样间隔 * 配置比例（50%）
    base_prediction_days = int(global_median_interval * config.MOGP_PREDICTION_WINDOW['ratio_of_interval'])

    # 应用安全约束 (最小7天/最大14天)
    prediction_days = int(max(
        config.MOGP_PREDICTION_WINDOW['min_days'],
        min(base_prediction_days, config.MOGP_PREDICTION_WINDOW['max_days'])
    ))

    logger.info(f"\n📊 MOGP 预测窗口最终决策:")
    logger.info(f"  - 基础窗口（50%中位间隔）: {base_prediction_days}天")
    logger.info(f"  - 应用约束后: {prediction_days}天")
    logger.info(f"  - 复查间隔提示: {int(global_median_interval)}天后")
    logger.info(f"  - 配置约束: 最小{config.MOGP_PREDICTION_WINDOW['min_days']}天，"
                f"最大{config.MOGP_PREDICTION_WINDOW['max_days']}天\n")
    
    # ========== 阶段7: 动态调整数值稳定性参数 ==========
    if data_density < 0.05:
        initial_jitter = 1e-1
        logger.warning(f"  MOGP 数值稳定性: ⚠️ 稀疏数据，使用增强 jitter={initial_jitter}")
    else:
        initial_jitter = 1e-2
        logger.info(f"  MOGP 数值稳定性: ✅ 密集数据，使用标准 jitter={initial_jitter}")

    gpytorch.settings.cholesky_jitter._global_float_value = initial_jitter
    
    # 提高 Likelihood 的最小噪声
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.Interval(1e-3, 1.0)
    )
    
    # ========== 阶段8: 初始化MOGP模型 ==========
    model = MultitaskGPModel(
        train_x, train_y, likelihood, num_tasks, 
        df_subset=df_subset,
        indicator_names=target_indicators
    )
    model = model.double()
    likelihood = likelihood.double()
    model.train()
    likelihood.train()

    # ========== 阶段9: 智能调整学习率 ==========
    if has_any_spike:
        if data_density < 0.05:
            linear_lr = 0.15
            logger.info(f"  MOGP: 增加LinearKernel学习率至0.15（稀疏数据+强趋势）")
        else:
            linear_lr = 0.3
            logger.info(f"  MOGP: 增加LinearKernel学习率至0.3（密集数据+强趋势）")
    else:
        linear_lr = 0.05
        logger.info(f"  MOGP: 使用标准LinearKernel学习率0.05")

    optimizer = torch.optim.Adam([
        {'params': model.covar_module.data_covar_module.kernels[0].parameters(), 'lr': 0.02},
        {'params': model.covar_module.data_covar_module.kernels[1].parameters(), 'lr': linear_lr},
        {'params': model.mean_module.parameters(), 'lr': 0.01},
        {'params': likelihood.parameters(), 'lr': 0.01}
    ])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ========== 阶段10: 动态计算训练迭代次数 ==========
    # 根据密度决定每点迭代次数
    if data_density > 0.1:  # 密集 (>1次/10天)
        iter_per_point = 25
        min_total = 200
    elif data_density > 0.05:  # 中等 (1次/10-20天)
        iter_per_point = 40
        min_total = 300
    else:  # 稀疏 (<1次/20天)
        iter_per_point = 60
        min_total = 400

    # 计算总迭代次数 (下限根据密度确定, 上限800)
    training_iter = int(max(min_total, min(800, len(df_subset) * iter_per_point)))
    logger.info(
        f"  MOGP(train): 设置最大迭代次数为 {training_iter} "
        f"(数据密度={data_density:.3f}, 每点={iter_per_point}次)"
    )

    # ========== 阶段11: 训练MOGP模型（带早停） ==========
    best_loss = float('inf')
    patience = 80
    no_improve_count = 0

    logger.info(f"  MOGP: 使用增强的数值稳定性配置 (全局 cholesky_jitter={initial_jitter})")
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if (i + 1) % 100 == 0:
            logger.debug(f"  MOGP(train): Iter {i+1}/{training_iter}, Loss: {loss.item():.3f}")

        if no_improve_count >= patience:
            logger.info(f"  MOGP(train): 早停于第 {i+1} 轮")
            break

    logger.info(f"  MOGP(train): 训练完成！最终Loss: {best_loss:.3f}")

    # ========== 阶段12: 输出学习到的Kernel参数 ==========
    logger.info(f"\n  MOGP: 学习到的Kernel参数:")
    with torch.no_grad():
        matern_lengthscale = model.covar_module.data_covar_module.kernels[0].base_kernel.lengthscale.item()
        linear_variance = model.covar_module.data_covar_module.kernels[1].outputscale.item()
        logger.info(f"  - Matern lengthscale: {matern_lengthscale:.1f} 天")
        logger.info(f"  - Linear variance: {linear_variance:.3f}")

    # ========== 阶段13: 生成GP预测（未来点） ==========
    logger.info(f"\n  MOGP: 开始生成智能预测（基于GP Kernel自动推断）...")

    model.eval()
    likelihood.eval()

    original_last_day = train_x.max().item()
    future_start_day = original_last_day
    future_end_day = original_last_day + prediction_days
    future_days_tensor = torch.linspace(
        future_start_day, future_end_day, prediction_days + 1, dtype=torch.float64
    ).unsqueeze(-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_prediction = likelihood(model(future_days_tensor))
        
        future_mean_scaled = gp_prediction.mean
        lower_scaled, upper_scaled = gp_prediction.confidence_region()
        future_std_scaled = (upper_scaled - lower_scaled) / (2 * 1.96)

    # ========== 阶段14: 修正T=0时刻的均值和不确定性 ==========
    logger.info("\n  MOGP: 对齐 T 时刻的预测均值与实际观测值 (未来点不受影响)")
    for i in range(num_tasks):
        gp_predicted_t0_scaled = future_mean_scaled[0, i].item()
        actual_t0_scaled = train_y[-1, i].item()  # 获取最后一个训练点的真实值

        # 只强制修正第一个预测点 (t=0) 的均值
        future_mean_scaled[0, i] = actual_t0_scaled

        logger.info(
            f"    - {target_indicators[i]} @ t=0: GP Mean={gp_predicted_t0_scaled:.3f}, "
            f"Actual={actual_t0_scaled:.3f} -> Corrected Mean={future_mean_scaled[0, i]:.3f}"
        )

    logger.info("  MOGP: 调整 t=0 时刻的不确定性以反映测量误差...")

    MEASUREMENT_ERROR_STD_RATIO = 0.05  # 假设5%的测量误差比例
    MIN_ABS_UNCERTAINTY = 0.05  # 最小绝对不确定性（缩放后）

    for i in range(num_tasks):
        actual_t0_scaled = train_y[-1, i].item()
        measurement_std = abs(actual_t0_scaled) * MEASUREMENT_ERROR_STD_RATIO
        # t=0 的标准差取 GP预测的标准差 和 基于测量的标准差 中的【较大值】
        future_std_scaled[0, i] = max(
            future_std_scaled[0, i].item(), 
            measurement_std, 
            MIN_ABS_UNCERTAINTY
        )
        logger.info(f"    - {target_indicators[i]} @ t=0: Final Std Dev = {future_std_scaled[0, i]:.3f}")

    logger.info("  MOGP: 预测完成")

    # ========== 阶段15: 构建参考范围字典 ==========
    reference_ranges = {}
    for report_type, indicators_list in config.LAB_REPORT_CONFIG.items():
        for indicator_info in indicators_list:
            indicator_name = indicator_info['name']
            lower_value = indicator_info.get('lower')
            upper_value = indicator_info.get('upper')
            
            reference_ranges[indicator_name] = {
                'lower': lower_value if lower_value is not None else 0,
                'upper': upper_value
            }
    
    for indicator in target_indicators:
        if indicator not in reference_ranges:
            reference_ranges[indicator] = {'lower': 0, 'upper': None}
        elif reference_ranges[indicator]['lower'] is None:
            reference_ranges[indicator]['lower'] = 0

    # ========== 阶段16: 定义辅助函数（统一的置信区间计算） ==========
    def calculate_lognormal_lower_bound(mean_val: torch.Tensor, std_val: torch.Tensor, 
                                       bio_lower: float, point_label: str = "") -> torch.Tensor:
        """
        【V4.8 新增】统一的对数正态下限计算函数
        
        :param mean_val: 均值（原始单位）
        :param std_val: 标准差（原始单位）
        :param bio_lower: 生物学下限
        :param point_label: 日志标签（例如 "未来点j=0" 或 "历史点j=3"）
        :return: 下限值
        """
        # 1. 均值太小，使用对称方法
        if mean_val <= 0.5:
            lower_bound = mean_val - 1.96 * std_val
            if point_label:
                logger.debug(f"    {point_label}: 均值过小({mean_val:.2f})，使用对称法")
            return max(lower_bound.item(), bio_lower + 0.01)
        
        # 2. 计算对数正态参数
        mu = torch.log(mean_val)
        sigma = std_val / (mean_val + 1e-6)
        
        # 3. 检查 sigma 合理性
        if not (0.01 < sigma.item() < 5.0):
            lower_fallback = mean_val - 1.96 * std_val
            if point_label:
                logger.debug(
                    f"    {point_label}: sigma不合理({sigma.item():.3f})，"
                    f"使用对称法（下限={lower_fallback.item():.2f}）"
                )
            return max(lower_fallback.item(), bio_lower + 0.01)
        
        # 4. 尝试对数正态计算
        try:
            lognorm_lower = lognorm.ppf(0.025, s=sigma.item(), scale=torch.exp(mu).item())
            return max(lognorm_lower, bio_lower + 0.01)
        except Exception as e:
            lower_fallback = mean_val - 1.96 * std_val
            if point_label:
                logger.warning(
                    f"    {point_label}: 对数正态计算异常 ({e})，"
                    f"回退到对称法（下限={lower_fallback.item():.2f}）"
                )
            return max(lower_fallback.item(), bio_lower + 0.01)

    # ========== 阶段17: 【核心】计算未来点的置信区间 ==========
    logger.info(f"\n 第一阶段：计算未来预测点的置信区间...")
    future_mean = torch.zeros_like(future_mean_scaled)
    future_std = torch.zeros_like(future_std_scaled)
    future_lower = torch.zeros_like(future_mean_scaled)
    future_upper = torch.zeros_like(future_mean_scaled)
    
    for i, indicator in enumerate(target_indicators):
        scaler = scalers[indicator]
        bio_lower = reference_ranges.get(indicator, {}).get('lower', 0)
        
        # 1. 逆标准化均值和标准差
        mean_original = scaler.inverse_transform(
            future_mean_scaled[:, i].numpy().reshape(-1, 1)
        ).flatten()
        future_mean[:, i] = torch.tensor(mean_original, dtype=torch.float64)

        std_original = future_std_scaled[:, i].numpy() * scaler.scale_[0]
        future_std[:, i] = torch.tensor(std_original, dtype=torch.float64)

        # 2. 计算上限（对称方法）
        upper_scaled_vals = future_mean_scaled[:, i] + 1.96 * future_std_scaled[:, i]
        upper_original = scaler.inverse_transform(
            upper_scaled_vals.numpy().reshape(-1, 1)
        ).flatten()
        future_upper[:, i] = torch.tensor(upper_original, dtype=torch.float64)

        # 3. 计算下限（使用统一的辅助函数）
        logger.debug(f"  计算 {indicator} 的未来下限（共{len(future_mean)}个点）:")
        for j in range(len(future_mean)):
            future_lower[j, i] = calculate_lognormal_lower_bound(
                future_mean[j, i], 
                future_std[j, i], 
                bio_lower,
                point_label=f"未来点j={j}"
            )

        logger.info(f"  - {indicator}:")
        logger.info(f"      T=0: {future_mean[0,i].item():.2f} ± {future_std[0,i].item():.2f}")
        logger.info(f"      +{prediction_days}天: {future_mean[-1,i].item():.2f} ± {future_std[-1,i].item():.2f}")

    # ========== 阶段18: 处理历史数据并强制对齐T-1/T=0 ==========
    logger.info(f"\n  第二阶段：处理历史数据并对齐 T-1/T=0...")
    
    historical_train_x = train_x
    historical_train_y = train_y
    historical_days = historical_train_x.squeeze()
    historical_true_values_scaled = historical_train_y
    
    hist_mean = torch.zeros_like(historical_true_values_scaled)
    hist_std = torch.zeros_like(historical_true_values_scaled)
    hist_lower = torch.zeros_like(historical_true_values_scaled)
    hist_upper = torch.zeros_like(historical_true_values_scaled)
    
    for i, indicator in enumerate(target_indicators):
        scaler = scalers[indicator]
        bio_lower = reference_ranges.get(indicator, {}).get('lower', 0)
        
        # 1. 逆标准化历史均值（真实值）
        mean_hist = scaler.inverse_transform(
            historical_true_values_scaled[:, i].numpy().reshape(-1, 1)
        ).flatten()
        hist_mean[:, i] = torch.tensor(mean_hist, dtype=torch.float64)

        # 2. 计算历史点标准差（排除最后一个点）
        for j in range(len(hist_mean) - 1):
            measurement_std = hist_mean[j, i] * MEASUREMENT_ERROR_STD_RATIO
            hist_std[j, i] = max(measurement_std, MIN_ABS_UNCERTAINTY)
        
        # 3. 【关键】T-1 点使用与 T=0 相同的标准差
        hist_std[-1, i] = future_std[0, i]
        
        # 4. 强制未来T时刻的均值等于历史最后值
        future_mean[0, i] = hist_mean[-1, i]
        
        # 5. 计算历史点置信区间（排除最后一个点）
        logger.debug(f"  计算 {indicator} 的历史下限（共{len(hist_mean)-1}个点）:")
        for j in range(len(hist_lower) - 1):
            # 上限（对称）
            hist_upper[j, i] = hist_mean[j, i] + 1.96 * hist_std[j, i]
            
            # 下限（统一使用辅助函数）
            hist_lower[j, i] = calculate_lognormal_lower_bound(
                hist_mean[j, i],
                hist_std[j, i],
                bio_lower,
                point_label=f"历史点j={j}"
            )
        
        # 6. 【核心修复】强制 T-1 与 T=0 的置信区间完全对齐
        hist_lower[-1, i] = future_lower[0, i]
        hist_upper[-1, i] = future_upper[0, i]
        
        # 7. 验证对齐（保留源代码的详细验证）
        t_minus_1_lower = hist_lower[-1, i].item()
        t_minus_1_upper = hist_upper[-1, i].item()
        t0_lower = future_lower[0, i].item()
        t0_upper = future_upper[0, i].item()
        
        logger.info(f"  - {indicator} 置信区间对齐:")
        logger.info(f"      T-1: [{t_minus_1_lower:.2f}, {t_minus_1_upper:.2f}]")
        logger.info(f"      T=0: [{t0_lower:.2f}, {t0_upper:.2f}]")
        
        alignment_gap_lower = abs(t_minus_1_lower - t0_lower)
        alignment_gap_upper = abs(t_minus_1_upper - t0_upper)
        
        if alignment_gap_lower < 0.01 and alignment_gap_upper < 0.01:
            logger.info(f"      ✅ 完全对齐，无跳空")
        else:
            logger.warning(
                f"      ⚠️ 对齐异常！下限间隙={alignment_gap_lower:.2f}, "
                f"上限间隙={alignment_gap_upper:.2f}"
            )

    # ========== 阶段19: 数据拼接（避免T=0重复） ==========
    future_days_inclusive = future_days_tensor.squeeze()
    
    # 避免 t=0 重复，历史数据不包含最后一个点
    all_days = torch.cat([historical_days[:-1], future_days_inclusive])
    all_mean = torch.cat([hist_mean[:-1], future_mean])
    all_lower = torch.cat([hist_lower[:-1], future_lower])
    all_upper = torch.cat([hist_upper[:-1], future_upper])
    all_std = torch.cat([hist_std[:-1], future_std])

    # ========== 阶段20: 计算趋势（基于合并后的数据） ==========
    all_trend = torch.zeros_like(all_mean)
    for i in range(num_tasks):
        # 1. 检查数据点数量
        if len(all_days) < 2:
            # 数据点不足，无法计算梯度，趋势设为0
            all_trend[:, i] = 0.0
            logger.debug(f"  ⚠️ {target_indicators[i]}: 数据点不足 (<2)，无法计算趋势。")
            continue

        # 2. 获取均值和对应的时间坐标（天数）
        mean_vals = all_mean[:, i].numpy()
        time_coords = all_days.numpy() # 直接使用 all_days (长度 N) 作为坐标

        # 3. 使用 np.gradient 计算趋势，直接传入坐标数组
        try:
            # 核心修改：将 time_coords 直接作为第二个参数传递给 np.gradient
            # NumPy 会自动处理非均匀间隔
            trend_vals = np.gradient(mean_vals, time_coords)

            # 4. 健壮性处理：替换计算中可能产生的 NaN 或 inf
            # np.gradient 在边界或数值精度问题下可能产生无效值
            trend_vals = np.nan_to_num(trend_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # 调试日志，检查计算结果范围
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"  {target_indicators[i]} 趋势计算 (np.gradient with coords): "
                    f"范围=[{trend_vals.min():.4f}, {trend_vals.max():.4f}]"
                )

        except Exception as e:
            # 捕获其他潜在错误（例如内存问题等）
            logger.warning(
                f"  ⚠️ {target_indicators[i]} 趋势计算时发生意外错误: {e}，使用0填充",
                exc_info=True # 记录详细错误堆栈
            )
            trend_vals = np.zeros_like(mean_vals)

        # 5. 将计算结果存回 tensor
        all_trend[:, i] = torch.tensor(trend_vals, dtype=torch.float64)

    # ========== 阶段21: 拼接验证 ==========
    logger.info(f"\n  MOGP: 数据拼接验证:")
    expected_total = len(historical_days) - 1 + len(future_days_inclusive)
    logger.info(f"    - 历史点数（不含最后）: {len(historical_days)-1}")
    logger.info(f"    - 未来点数（含T=0）: {len(future_days_inclusive)}")
    logger.info(f"    - 实际总点数: {len(all_days)}, 预期: {expected_total}")
    
    if len(all_days) != expected_total:
        logger.error(f"❌ 拼接失败！")
        raise ValueError("MOGP 数据拼接失败：点数不匹配")

    # ========== 阶段22: 调试日志（边界点检查） ==========
    for i, indicator in enumerate(target_indicators):
        logger.debug(f"\n    MOGP_DEBUG: {indicator} 边界点检查:")
        if len(hist_mean) >= 3:
            logger.debug(f"    历史[-3]: mean={hist_mean[-3,i]:.2f}, CI=[{hist_lower[-3,i]:.2f}, {hist_upper[-3,i]:.2f}]")
            logger.debug(f"    历史[-2]: mean={hist_mean[-2,i]:.2f}, CI=[{hist_lower[-2,i]:.2f}, {hist_upper[-2,i]:.2f}]")
        logger.debug(f"    历史[-1]: mean={hist_mean[-1,i]:.2f}, CI=[{hist_lower[-1,i]:.2f}, {hist_upper[-1,i]:.2f}] ← 已删除")
        logger.debug(f"    未来[0]: mean={future_mean[0,i]:.2f}, CI=[{future_lower[0,i]:.2f}, {future_upper[0,i]:.2f}] ← 已使用")
        if len(future_mean) > 1:
            logger.debug(f"    未来[1]: mean={future_mean[1,i]:.2f}, CI=[{future_lower[1,i]:.2f}, {future_upper[1,i]:.2f}]")

    # ========== 阶段23: 构建返回结果字典 ==========
    results = {}
    all_dates = [first_day + pd.Timedelta(days=int(d)) for d in all_days.numpy()]
    historical_dates = df_subset.index.tolist()
    future_dates = [first_day + pd.Timedelta(days=int(d)) for d in future_days_inclusive.numpy()]

    for i, indicator in enumerate(target_indicators):
        results[indicator] = {
            'historical_dates': historical_dates,
            'historical_values': df_subset[indicator].values,
            'all_dates': all_dates,
            'all_predicted_mean': all_mean[:, i].numpy(),
            'all_predicted_std': all_std[:, i].numpy(),
            'all_confidence_lower': all_lower[:, i].numpy(),
            'all_confidence_upper': all_upper[:, i].numpy(),
            'all_uncertainty': all_std[:, i].numpy(),
            'all_trend': all_trend[:, i].numpy(),
            'future_dates': future_dates,
            'predicted_mean': future_mean[:, i].numpy(),
            'confidence_lower': future_lower[:, i].numpy(),
            'confidence_upper': future_upper[:, i].numpy(),
            'next_check_days': int(global_median_interval),
            'confidence': 'medium',
            'warning': None
        }
    
    # ========== 阶段24: 残差诊断 ==========
    logger.info("\n" + "="*80)
    logger.info("📊 模型诊断分析")
    logger.info("="*80)

    for i, indicator in enumerate(target_indicators):
        # 使用历史数据进行残差诊断
        true_vals = df_subset[indicator].values
        
        # 获取模型对历史数据的拟合值
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            hist_pred = likelihood(model(historical_train_x))
            fitted_scaled = hist_pred.mean[:, i].numpy()
        
        # 逆标准化
        fitted_values = scalers[indicator].inverse_transform(
            fitted_scaled.reshape(-1, 1)
        ).flatten()
        
        # 残差诊断
        diagnostics = residual_diagnostics(true_vals, fitted_values, indicator)
        
        # 存储诊断结果
        if 'diagnostics' not in results[indicator]:
            results[indicator]['diagnostics'] = {}
        results[indicator]['diagnostics'] = diagnostics
    
    # ========== 阶段25: 可靠性评估（临床数据适配版） ==========
    logger.info("\n" + "="*80)
    logger.info("🎯 可靠性评估（临床数据适配版）")
    logger.info("="*80)
    
    # 计算全局数据密度指标
    avg_interval = time_span_days / max(1, len(train_x) - 1)
    
    logger.info(f"\n数据密度分析:")
    logger.info(f"  - 时间跨度: {time_span_days} 天")
    logger.info(f"  - 数据点数: {len(train_x)} 个")
    logger.info(f"  - 平均采样间隔: {avg_interval:.0f} 天/次")
    
    # 根据采样间隔动态调整阈值
    if avg_interval > 60:  # 稀疏数据（>60天/次）
        logger.info(f"✅ 检测到临床复查频率（{avg_interval:.0f}天/次），启用宽松判定模式")
        
        # 动态放宽系数：最多3倍
        relaxation_factor = min(3.0, avg_interval / 30.0)
        
        uncertainty_threshold = (
            config.MOGP_CONFIDENCE_THRESHOLDS['uncertainty_growth_high'] * relaxation_factor
        )
        error_threshold = (
            config.MOGP_CONFIDENCE_THRESHOLDS['error_ratio_high'] * relaxation_factor
        )
        
        logger.info(
            f"阈值调整: "
            f"不确定性增长阈值 {uncertainty_threshold:.2f} (×{relaxation_factor:.1f}), "
            f"拟合误差阈值 {error_threshold:.2f} (×{relaxation_factor:.1f})"
        )
    else:
        # 密集数据使用原始阈值
        uncertainty_threshold = config.MOGP_CONFIDENCE_THRESHOLDS['uncertainty_growth_high']
        error_threshold = config.MOGP_CONFIDENCE_THRESHOLDS['error_ratio_high']
        logger.info(f"使用标准阈值（数据密度充足，平均 {avg_interval:.0f} 天/次）")
    
    # 对每个指标进行可靠性评估
    for i, indicator in enumerate(target_indicators):
        logger.info(f"\n--- {indicator} 可靠性评估 ---")
        
        # 1. 计算不确定性增长率
        initial_std = future_std[0, i].item()  # T=0 时刻的标准差
        final_std = future_std[-1, i].item()   # 预测终点的标准差
        
        if initial_std > 1e-6:
            uncertainty_growth_ratio = (final_std - initial_std) / initial_std
        else:
            uncertainty_growth_ratio = 0.0
        
        # 2. 计算历史拟合误差比
        true_vals = df_subset[indicator].values
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            hist_pred = likelihood(model(historical_train_x))
            fitted_scaled = hist_pred.mean[:, i].numpy()
        
        fitted_values = scalers[indicator].inverse_transform(
            fitted_scaled.reshape(-1, 1)
        ).flatten()
        
        residuals = true_vals - fitted_values
        rmse = np.sqrt(np.mean(residuals**2))
        hist_std = np.std(true_vals)
        
        if hist_std > 1e-6:
            error_ratio = rmse / hist_std
        else:
            error_ratio = 0.0
        
        # 3. 使用动态阈值判定可靠性
        if uncertainty_growth_ratio > uncertainty_threshold:
            confidence = 'low'
            warning = (
                f'⚠️ 不确定性增长 {uncertainty_growth_ratio:.1%}（阈值 {uncertainty_threshold:.1%}），'
                f'预测仅供参考'
            )
        elif error_ratio > error_threshold:
            confidence = 'medium'
            warning = (
                f'ℹ️ 拟合误差 {error_ratio:.2f} 倍标准差（阈值 {error_threshold:.1f}），'
                f'请结合基线观察'
            )
        else:
            confidence = 'high'
            warning = None
        
        # 4. 记录详细日志
        logger.info(
            f"  可靠性: {confidence.upper()} | "
            f"不确定性增长: {uncertainty_growth_ratio:.1%} (阈值: {uncertainty_threshold:.1%}) | "
            f"拟合误差比: {error_ratio:.2f} (阈值: {error_threshold:.1f})"
        )
        
        if warning:
            logger.info(f"  警告: {warning}")
        else:
            logger.info(f"  ✅ 预测可信度高，趋势和数值均可参考")
        
        # 5. 更新结果字典
        results[indicator]['confidence'] = confidence
        results[indicator]['warning'] = warning
        results[indicator]['confidence_metrics'] = {
            'uncertainty_growth_ratio': uncertainty_growth_ratio,
            'uncertainty_threshold': uncertainty_threshold,
            'error_ratio': error_ratio,
            'error_threshold': error_threshold,
            'avg_sampling_interval': avg_interval,
            'relaxation_factor': relaxation_factor if avg_interval > 60 else 1.0
        }
    
    logger.info("="*80 + "\n")

    # ========== 阶段26: 交叉验证（如果数据点>=5） ==========
    if len(df_subset) >= 5:
        cv_results = leave_one_out_cross_validation(
            df_subset, target_indicators, scalers, num_tasks
        )
        
        # 存储交叉验证结果
        for indicator in target_indicators:
            if cv_results[indicator] is not None:
                results[indicator]['cv_metrics'] = cv_results[indicator]
    else:
        logger.warning("\n⚠️ MOGP: 数据点不足5个，跳过交叉验证")

    logger.info("="*80 + "\n")

    # ========== 阶段27: 返回结果 ==========
    logger.info(f"✅ 智能预测完成！")
    logger.info("="*80 + "\n")

    return results


def find_most_similar_pattern_dtw(
    query_series: pd.Series, 
    history_series: pd.Series, 
    snippet_len: int,
    time_tolerance_ratio: float = 0.35
):
    """
    【V5.5 全范围匹配版】
    使用DTW在历史中寻找最相似模式，支持自适应容忍度和全范围兜底。
    
    核心改进：
    1. 检测采样密度的不均匀性（变异系数）
    2. 检测 Query 与历史中位数的偏离
    3. 自动放宽时间容忍度（从35% → 80% → 100%）
    4. 对于极端不均匀数据，使用绝对范围而非相对范围
    5. 🔥 新增：当所有策略失败时，使用全历史范围兜底
    """
    
    logger.info(f"开始 DTW 相似模式查找 (V5.5 全范围版)... (Query len: {snippet_len}, History len: {len(history_series)})")
    
    # --- 1. 输入验证 ---
    if len(query_series) != snippet_len:
        logger.warning(f"DTW: Query 长度不匹配。期望={snippet_len}, 实际={len(query_series)}")
        return None, float('inf')
    
    if len(history_series) < snippet_len * 2:
        logger.warning(
            f"DTW: 历史数据不足。History={len(history_series)}, "
            f"需要至少 {snippet_len * 2} 个点（2倍片段长度）"
        )
        return None, float('inf')
        
    if query_series.index.nunique() < 2:
        logger.warning("DTW: Query 序列时间点不足2，无法计算跨度。")
        return None, float('inf')

    # --- 2. 计算 Query 的时间跨度 ---
    query_duration_days = (query_series.index.max() - query_series.index.min()).days
    query_duration_days = max(1.0, float(query_duration_days))
    
    # --- 3. 采样密度分析 ---
    search_space = history_series.iloc[:-snippet_len]
    
    if len(search_space) < snippet_len:
        logger.warning(
            f"DTW: 排除 query 片段后，搜索空间不足。"
            f"Search space={len(search_space)}, Snippet={snippet_len}"
        )
        return None, float('inf')
    
    # 计算所有历史片段的时间跨度
    all_spans = []
    total_search_indices = len(search_space) - snippet_len + 1
    
    for i in range(total_search_indices):
        snippet = search_space.iloc[i : i + snippet_len]
        if snippet.index.nunique() < 2:
            continue
        span = (snippet.index.max() - snippet.index.min()).days
        all_spans.append(max(1.0, float(span)))
    
    if not all_spans:
        logger.warning("DTW: 没有有效的历史片段（所有片段日期重复）。")
        return None, float('inf')
    
    # 计算统计特征
    median_span = np.median(all_spans)
    span_std = np.std(all_spans)
    span_cv = span_std / median_span if median_span > 0 else 1.0
    span_range_ratio = max(all_spans) / min(all_spans) if min(all_spans) > 0 else float('inf')
    query_deviation = abs(query_duration_days - median_span) / median_span if median_span > 0 else 1.0
    
    logger.info(f"\n  📊 DTW 采样密度分析:")
    logger.info(f"    - Query 跨度: {query_duration_days:.0f} 天")
    logger.info(f"    - 历史跨度范围: [{min(all_spans):.0f}, {max(all_spans):.0f}] 天")
    logger.info(f"    - 历史跨度中位数: {median_span:.0f} 天")
    logger.info(f"    - 变异系数 (CV): {span_cv:.2f}")
    logger.info(f"    - 跨度范围比: {span_range_ratio:.1f}倍")
    logger.info(f"    - Query偏离度: {query_deviation:.1%}")
    
    # --- 4. 自适应策略 ---
    original_tolerance = time_tolerance_ratio
    use_absolute_range = False
    use_full_range_fallback = False  # 🔥 新增标志
    
    # 策略1: CV > 0.5
    if span_cv > 0.5:
        time_tolerance_ratio = max(0.80, time_tolerance_ratio)
        logger.info(
            f"  🔧 DTW 自适应: 检测到不均匀采样（CV={span_cv:.2f}），"
            f"容忍度从 {original_tolerance:.0%} 放宽到 {time_tolerance_ratio:.0%}"
        )
    
    # 策略2: Query偏离 > 50%
    if query_deviation > 0.5:
        time_tolerance_ratio = max(1.00, time_tolerance_ratio)
        logger.info(
            f"  🔧 DTW 自适应: Query 跨度（{query_duration_days:.0f}天）与历史中位数（{median_span:.0f}天）"
            f"差异较大（{query_deviation:.0%}），容忍度进一步放宽到 {time_tolerance_ratio:.0%}"
        )
    
    # 策略3: 跨度范围比 > 3
    if span_range_ratio > 3.0:
        use_absolute_range = True
        percentile_10 = np.percentile(all_spans, 10)
        percentile_90 = np.percentile(all_spans, 90)
        min_allowed_duration = percentile_10
        max_allowed_duration = percentile_90
        
        logger.info(
            f"  🔧 DTW 自适应: 历史跨度范围极大（{span_range_ratio:.1f}倍），"
            f"改用绝对范围 [{min_allowed_duration:.0f}, {max_allowed_duration:.0f}] 天"
        )
    else:
        min_allowed_duration = query_duration_days * (1.0 - time_tolerance_ratio)
        max_allowed_duration = query_duration_days * (1.0 + time_tolerance_ratio)
        logger.info(
            f"  ✅ DTW: 使用相对范围匹配（容忍度={time_tolerance_ratio:.0%}），"
            f"允许范围: [{min_allowed_duration:.0f}, {max_allowed_duration:.0f}] 天"
        )
    
    # --- 5. 滑动窗口生成 ---
    historical_snippets = []
    
    for i in range(total_search_indices):
        snippet = search_space.iloc[i : i + snippet_len]
        
        if snippet.index.nunique() < 2:
            continue
            
        snippet_duration_days = (snippet.index.max() - snippet.index.min()).days
        snippet_duration_days = max(1.0, float(snippet_duration_days))

        if min_allowed_duration <= snippet_duration_days <= max_allowed_duration:
            historical_snippets.append(snippet)
    
    # 🔥 策略4: 全范围兜底（当没有任何片段匹配时）
    if not historical_snippets:
        logger.warning(
            f"  ⚠️ DTW: 未能在调整后的容忍度内找到匹配的历史片段。"
            f"\n      - 允许范围: [{min_allowed_duration:.0f}, {max_allowed_duration:.0f}] 天"
            f"\n      - 历史跨度: [{min(all_spans):.0f}, {max(all_spans):.0f}] 天"
        )
        
        # 启用全范围兜底策略
        use_full_range_fallback = True
        min_allowed_duration = min(all_spans) * 0.8  # 留 20% 安全边界
        max_allowed_duration = max(all_spans) * 1.2
        
        logger.info(
            f"  🔥 DTW 兜底策略: 改用全历史范围匹配"
            f"\n      - 新允许范围: [{min_allowed_duration:.0f}, {max_allowed_duration:.0f}] 天"
            f"\n      - 这将匹配所有 {total_search_indices} 个候选片段"
        )
        
        # 重新生成片段列表（使用全范围）
        for i in range(total_search_indices):
            snippet = search_space.iloc[i : i + snippet_len]
            
            if snippet.index.nunique() < 2:
                continue
                
            snippet_duration_days = (snippet.index.max() - snippet.index.min()).days
            snippet_duration_days = max(1.0, float(snippet_duration_days))

            if min_allowed_duration <= snippet_duration_days <= max_allowed_duration:
                historical_snippets.append(snippet)
    
    # 最终检查
    if not historical_snippets:
        logger.error(
            f"  ❌ DTW: 即使使用全范围兜底策略，仍未找到有效片段。"
            f"\n      提示: 检查数据质量或增加数据点"
        )
        return None, float('inf')
    
    logger.info(
        f"  ✅ DTW: 在 {total_search_indices} 个候选片段中，"
        f"找到了 {len(historical_snippets)} 个符合条件的片段"
        f"{' (使用兜底策略)' if use_full_range_fallback else ''}"
    )
    
    # --- 6. 距离计算 ---
    query_values = query_series.values.astype(float)
    distances = [
        dtw.distance_fast(query_values, snippet.values.astype(float))
        for snippet in historical_snippets
    ]
    
    if not distances:
        logger.warning("DTW: 距离计算列表为空。")
        return None, float('inf')
        
    # --- 7. 结果验证 ---
    best_match_index = np.argmin(distances)
    min_distance = distances[best_match_index]
    most_similar_snippet = historical_snippets[best_match_index]
    
    # 未来泄漏检查
    if most_similar_snippet.index[-1] >= query_series.index[0]:
        logger.error(
            f"  ❌ DTW: 检测到时间重叠！"
            f"匹配片段结束于 {most_similar_snippet.index[-1]}, "
            f"查询片段开始于 {query_series.index[0]}"
        )
        return None, float('inf')
    
    matched_span = (most_similar_snippet.index.max() - most_similar_snippet.index.min()).days
    
    logger.info(
        f"  ✅ DTW 查找完成！"
        f"\n    - 最小距离: {min_distance:.2f}"
        f"\n    - 匹配片段日期: {most_similar_snippet.index.min().strftime('%Y-%m-%d')} "
        f"至 {most_similar_snippet.index[-1].strftime('%Y-%m-%d')}"
        f"\n    - 匹配片段跨度: {matched_span:.0f} 天 (Query: {query_duration_days:.0f} 天)"
        f"\n    - 应用策略: {'全范围兜底' if use_full_range_fallback else ('绝对范围' if use_absolute_range else '相对范围')}"
    )
    
    return most_similar_snippet, min_distance


def train_complete_pipeline(df: pd.DataFrame, feature_columns: list, patient_id: int):
    """
    一键式完整训练管道。
    1. 调用 discover_dynamic_indicators 获取【最终权重】和【调权详情】。
    2. 使用【最终权重】来训练【双相】健康基线模型。
    
    :param df: 包含所有阶段数据的完整DataFrame。
    :param feature_columns: 用于训练的特征列名称列表。
    :param patient_id: 当前病人的ID。
    :return: 一个元组 (models_dict, final_weights)，成功则包含模型和权重，失败则为 (None, None)。
    """

    logger.info("===== 开始执行完整训练管道 (train_complete_pipeline) =====")
    
    # 步骤 1: 调用新的高阶调权策略
    context_columns = [col for col in df.columns if col not in feature_columns]
    
    final_weights, boost_df, dynamic_params, fallback_activated = discover_dynamic_indicators(
        df, 
        feature_columns, 
        context_columns, 
        patient_id
    )
    
    if fallback_activated:
        logger.warning("train_complete_pipeline: 激活了回退模式，将使用默认权重（均为1.0）。")
    else:
        logger.info("动态指标发现与调权成功。")
        
    # 步骤 2: 训练【双相】健康基线模型 (使用 final_weights)
    logger.info("开始训练【双相】健康基线模型 (使用最终权重)...")
    models_dict = train_baseline_model(df, feature_columns, final_weights) 
    
    if models_dict:
        logger.info("✅ 基线模型训练成功。")
        logger.info("===== 完整训练管道执行完毕 =====")
        models_dict['boost_df'] = boost_df 
        models_dict['dynamic_params'] = dynamic_params
        models_dict['fallback_activated'] = fallback_activated
        return models_dict, final_weights 
    else:
        logger.error("基线模型训练失败（所有阶段数据均不足）。")
        logger.info("===== 完整训练管道执行失败 =====")
        return None, None
