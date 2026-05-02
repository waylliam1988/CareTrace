# baseline_monitor.py
"""
个人基线监测模块 - 检测当前值是否偏离个人历史基线 (V6.0 融合版)
使用鲁棒统计（时间加权中位数/MAD）和动态调整参数
"""

import numpy as np
import pandas as pd
import logging
import explainability
import data_manager
import config # 导入配置

logger = logging.getLogger(__name__)

class PersonalBaselineMonitor:
    """
    个人化基线跟踪器 (V6.0)
    - 使用时间加权鲁棒统计
    - 动态校准半衰期
    - 根据数据波动性微调Z-score阈值
    """

    def __init__(self, patient_data: pd.DataFrame = None):
        """
        初始化监测器，并根据传入数据（如果提供）校准参数。

        :param patient_data: 可选，用于初始校准的患者数据
        """
        logger.info("初始化 PersonalBaselineMonitor...")
        # --- 1. 设置默认参数 ---
        self.half_life = config.DEFAULT_HALF_LIFE_DAYS
        self.z_thresholds = config.BASELINE_Z_SCORE_THRESHOLDS.copy()
        self.last_calibrated_patient_id = None
        self.last_data_hash = None

        # --- 2. 如果提供了数据，尝试初始校准 ---
        if patient_data is not None and not patient_data.empty:
            try:
                self.calibrate(patient_data)
            except Exception as e:
                logger.error(f"初始化期间校准失败: {e}", exc_info=True)
                logger.warning("将使用默认参数。")
        else:
            logger.info("未提供初始数据，使用默认参数。")

        logger.info(f"初始化完成。半衰期: {self.half_life:.0f}天, Z阈值(高): {self.z_thresholds['high']:.2f}")

    def calibrate(self, patient_data: pd.DataFrame):
        """
        根据患者数据动态校准半衰期和Z-score阈值。

        :param patient_data: 患者数据DataFrame (需要包含 'phase' 列和 DatetimeIndex)
        """
        logger.info("开始校准基线监测器参数...")
        if patient_data.empty or len(patient_data) < 5:
            logger.warning("数据不足 (<5)，无法校准，将保留当前参数。")
            return

        # --- a. 校准时间半衰期 ---
        try:
            intervals = patient_data.index.to_series().diff().dt.days.dropna()
            if len(intervals) >= 2:
                median_interval = intervals.median()
                # 策略：半衰期 = 3倍中位采样间隔
                new_half_life = median_interval * 3.0
                # 应用边界保护
                min_hl, max_hl = config.HALF_LIFE_BOUNDS
                self.half_life = max(min_hl, min(new_half_life, max_hl))
                logger.info(f"  校准半衰期: {self.half_life:.0f}天 (基于中位间隔: {median_interval:.0f}天)")
            else:
                logger.debug("  采样间隔数据不足，半衰期保持不变。")
        except Exception as e:
            logger.error(f"  校准半衰期失败: {e}", exc_info=True)

        # --- b. 校准 Z-Score 阈值 (基于稳定期波动性) ---
        try:
            if 'phase' in patient_data.columns:
                stable_data = patient_data[patient_data['phase'] == '稳定监控期']
                if len(stable_data) >= 5:
                    numeric_cols = stable_data.select_dtypes(include=np.number).columns.drop('report_uuid', errors='ignore')
                    cvs = []
                    for col in numeric_cols:
                        values = stable_data[col].dropna()
                        if len(values) >= 3: # 至少需要3个点计算CV
                            mean_val = values.mean()
                            std_val = values.std()
                            if abs(mean_val) > 1e-6:
                                cvs.append(std_val / mean_val)

                    if cvs:
                        median_cv = np.median(cvs)
                        logger.debug(f"  稳定期中位变异系数 (Median CV): {median_cv:.3f}")
                        # 波动大 -> 阈值略微放宽
                        if median_cv > config.BASELINE_CV_THRESHOLD_FOR_ADJUSTMENT:
                            factor = config.BASELINE_Z_SCORE_ADJUSTMENT_FACTOR
                            self.z_thresholds['high'] = config.BASELINE_Z_SCORE_THRESHOLDS['high'] * factor
                            self.z_thresholds['medium'] = config.BASELINE_Z_SCORE_THRESHOLDS['medium'] * factor
                            self.z_thresholds['low'] = config.BASELINE_Z_SCORE_THRESHOLDS['low'] * factor
                            logger.info(f"  数据波动性较大 (CV > {config.BASELINE_CV_THRESHOLD_FOR_ADJUSTMENT})，"
                                        f"Z-Score阈值已调整 (High: {self.z_thresholds['high']:.2f})")
                        else:
                            # 波动不大，使用默认阈值
                            self.z_thresholds = config.BASELINE_Z_SCORE_THRESHOLDS.copy()
                            logger.info("  数据波动性正常，使用默认Z-Score阈值。")
                else:
                    logger.debug("  稳定期数据不足 (<5)，Z-Score阈值保持不变。")
            else:
                logger.debug("  缺少 'phase' 列，Z-Score阈值保持不变。")
        except Exception as e:
            logger.error(f"  校准Z-Score阈值失败: {e}", exc_info=True)

        logger.info("参数校准完成。")

    def assess_current_value(
        self,
        patient_data: pd.DataFrame,
        indicator: str,
        ref_ranges: dict
    ) -> dict:
        """
        评估当前值相对于个人基线的偏离程度。

        :param patient_data: 完整的患者数据（包含phase列和DatetimeIndex）
        :param indicator: 指标名称
        :param ref_ranges: 参考范围字典 {indicator: (lower, upper)}
        :return: 评估结果字典
        """
        logger.debug(f"开始评估基线偏离: {indicator}")
        if indicator not in patient_data.columns:
            logger.error(f"  指标 '{indicator}' 在数据中不存在")
            return {'status': 'error', 'message': '指标不存在'}

        # --- 1. 获取当前数据点信息 ---
        try:
            latest_row = patient_data.iloc[-1]
            current_phase = latest_row['phase']
            current_value = latest_row[indicator]
            current_date = latest_row.name # Index is DatetimeIndex
        except IndexError:
             logger.error("  无法获取最新数据行")
             return {'status': 'error', 'message': '数据为空'}
        except KeyError as e:
            logger.error(f"  获取当前信息时出错: 缺少列 {e}")
            return {'status': 'error', 'message': f'缺少必要列: {e}'}

        if pd.isna(current_value):
            logger.debug(f"  当前值 '{indicator}' 为空，跳过评估")
            return {'status': 'no_value', 'message': '当前值缺失'}

        # --- 2. 获取同阶段的历史数据 ---
        phase_history = patient_data[
            (patient_data['phase'] == current_phase) &
            (patient_data.index < current_date) # 严格排除当前点
        ]
        marker_history = phase_history[indicator].dropna()

        if len(marker_history) < 3:
            logger.debug(f"  '{current_phase}' 历史数据不足 ({len(marker_history)} < 3)，无法计算基线")
            return {
                'status': 'insufficient_data',
                'message': f'{current_phase}历史数据不足(<3次)',
                'level': 'info' # 这是一个信息，不是错误
            }

        # --- 3. 计算时间加权基线 ---
        try:
            baseline_stats = self._calc_weighted_baseline(marker_history, current_date)
            if not baseline_stats: # 可能内部计算失败
                logger.warning("  计算加权基线失败")
                return {'status': 'error', 'message': '计算基线失败'}
        except Exception as e:
            logger.error(f"  计算加权基线时发生意外错误: {e}", exc_info=True)
            return {'status': 'error', 'message': '计算基线时出错'}

        # --- 4. 计算偏离程度 ---
        deviation = self._calc_deviation(current_value, baseline_stats)

        # --- 5. 临床判断 ---
        clinical_assessment = self._clinical_interpretation(
            current_value,                              # 传入当前值用于与参考范围比较
            deviation,                                  # 传入偏离度字典
            indicator,                                  # 传入指标名称
            ref_ranges.get(indicator, (None, None)),    # 传入参考范围
            baseline_stats                              # 传入基线统计字典
        )

        logger.debug(f"  评估完成: {indicator}, Level: {clinical_assessment.get('level', 'N/A')}, Z-Score: {deviation.get('modified_z_score', 0):.2f}")
        return {
            'status': 'success',
            **clinical_assessment,
            'baseline': baseline_stats,
            'current_value': current_value,
            'deviation': deviation
        }

    def _calc_weighted_baseline(
        self,
        indicator_history: pd.Series,
        current_date: pd.Timestamp
    ) -> dict | None:
        """计算时间加权的基线统计量 (中位数, MAD)"""
        if len(indicator_history) < 1: # 至少需要一个历史点
             return None

        values = indicator_history.values
        dates = indicator_history.index

        if len(values) == 0 or len(dates) == 0:
            logger.warning("  indicator_history 为空，无法计算基线")
            return None

        # 计算时间权重
        days_ago = (current_date - dates).days.values
        # 避免 days_ago < 0 (虽然理论上不应发生)
        days_ago = np.maximum(0, days_ago)
        time_weights = 0.5 ** (days_ago / self.half_life)
        # 避免权重和为0
        if time_weights.sum() < 1e-9:
             logger.warning("  所有时间权重接近于0，使用等权重")
             time_weights = np.ones_like(days_ago) / len(days_ago)
        else:
             time_weights /= time_weights.sum() # 归一化

        # 加权中位数
        weighted_median = self._weighted_quantile(values, time_weights, 0.5)

        # 加权MAD
        absolute_deviations = np.abs(values - weighted_median)
        weighted_mad = self._weighted_quantile(absolute_deviations, time_weights, 0.5)

        # 计算加权 IQR（四分位距）
        q25 = self._weighted_quantile(values, time_weights, 0.25)
        q75 = self._weighted_quantile(values, time_weights, 0.75)
        weighted_iqr = q75 - q25
        
        logger.debug(f"  基线统计: Median={weighted_median:.3f}, MAD={weighted_mad:.6f}, IQR={weighted_iqr:.6f}")

        # 增强 MAD=0 时的处理逻辑
        if weighted_mad < 1e-6 and len(values) >= 2:
            if weighted_iqr > 1e-6:
                # 使用 IQR 估算 MAD
                robust_std_approx = weighted_iqr / 1.349
                weighted_mad = robust_std_approx * 0.6745
                logger.debug(f"  MAD接近于0，使用 IQR 估算: {weighted_mad:.6f}")
            else:
                logger.warning("  MAD 和 IQR 均接近于0，历史数据可能无波动")
                # MAD 保持接近 0 是合理的（会在 _calc_deviation 中用 fallback 处理）

        return {
            'median': weighted_median,
            'mad': weighted_mad,
            'iqr': weighted_iqr,  # 返回 IQR
            'n_samples': len(values),
            'time_span_days': days_ago.max()
        }

    def _weighted_quantile(self, values, weights, q):
        """计算加权分位数 (确保处理NaN和空值)"""
        # 清理数据
        valid_mask = ~np.isnan(values) & ~np.isnan(weights)
        values = values[valid_mask]
        weights = weights[valid_mask]

        if len(values) == 0:
            return np.nan # 或者返回一个合理的默认值

        # 排序
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # 计算累积权重
        cum_weights = np.cumsum(sorted_weights)
        # 防止除零
        total_weight = cum_weights[-1]
        if total_weight < 1e-9:
            # 如果总权重为0，无法计算分位数，返回简单中位数或NaN
            logger.warning("  加权分位数计算：总权重为0")
            return np.median(values) if len(values) > 0 else np.nan

        cum_weights /= total_weight

        # 线性插值找到分位数
        return np.interp(q, cum_weights, sorted_values)


    def _calc_deviation(self, current_value, baseline_stats):
        """
        计算偏离程度（使用修正Z-score）
        
        增强除零保护，使用自适应 epsilon 和 IQR 回退策略
        """
        median = baseline_stats['median']
        mad = baseline_stats['mad']
        
        # 使用更大的 epsilon（从 1e-9 提升到 1e-6）
        # 这对于接近零的指标（如某些肿瘤标志物）更稳健
        safe_epsilon = max(1e-6, abs(median) * 0.001)  # 动态 epsilon，至少为中位数的 0.1%
        
        # 当 MAD 过小时，使用 IQR 回退策略
        if mad < safe_epsilon:
            logger.debug(f"  MAD 过小 ({mad:.6f})，尝试使用 IQR 回退策略")
            
            # 从 baseline_stats 获取 IQR（如果有）
            iqr = baseline_stats.get('iqr', 0)
            
            if iqr > safe_epsilon:
                # IQR 转 MAD：MAD ≈ IQR / 1.349（正态分布假设）
                mad_from_iqr = iqr / 1.349
                logger.debug(f"  使用 IQR 估算 MAD: {mad_from_iqr:.6f}")
                mad = mad_from_iqr
            else:
                # 如果 IQR 也为 0，使用回退值（中位数的 1%）
                fallback_mad = abs(median) * 0.01 if abs(median) > safe_epsilon else 0.01
                logger.warning(
                    f"  MAD 和 IQR 均过小，使用回退值: {fallback_mad:.6f} "
                    f"(历史数据可能无波动)"
                )
                mad = fallback_mad
        
        # 核心计算：修正Z-score
        # 使用增强后的 safe_epsilon
        modified_z = 0.6745 * (current_value - median) / (mad + safe_epsilon)
        
        # 百分比变化（相对于中位数基线）
        pct_change = (current_value - median) / (abs(median) + safe_epsilon)
        
        return {
            'modified_z_score': modified_z,
            'pct_change': pct_change,
            'absolute_diff': current_value - median
        }


    def _clinical_interpretation(
        self,
        current_value: float,
        deviation: dict,
        indicator: str,
        ref_range: tuple,
        baseline_stats: dict
    ) -> dict:
        """临床解读 (使用 self.z_thresholds)"""
        z_score = deviation['modified_z_score']
        pct_change = deviation['pct_change']
        abs_z = abs(z_score)

        # 使用校准后的阈值
        z_high = self.z_thresholds['high']
        z_medium = self.z_thresholds['medium']
        z_low = self.z_thresholds['low']

        # 判断级别
        if abs_z > z_high:
            level = 'high'
            severity = '显著偏离'
            recommendation = '观察提示：该偏离较大，请尽快咨询医生是否需要复查或影像学检查'
        elif abs_z > z_medium:
            level = 'medium'
            severity = '中度偏离'
            recommendation = '观察提示：可咨询医生是否需要提前复查'
        elif abs_z > z_low:
            level = 'low'
            severity = '轻度偏离'
            recommendation = '观察提示：继续观察趋势，复查安排请遵医嘱'
        else:
            level = 'normal'
            severity = '正常波动'
            recommendation = '观察提示：按医生计划复查'

        # 生成人类可读的解释
        direction = "升高" if z_score > 0 else ("降低" if z_score < 0 else "持平")
        explanation = (
            f"{indicator}当前值(<b>{current_value:.2f}</b>)相比个人基线(中位数≈{baseline_stats['median']:.2f})"
            f"{direction}了<b>{abs(pct_change)*100:.1f}%</b>，"
            f"偏离程度达到<b>{abs_z:.1f}倍</b>标准差(修正Z-score)"
        )

        if level == 'high':
            explanation += "。<span style='color:red;'>这是一个显著的变化，需要引起重视。</span>"
        elif level == 'medium':
            explanation += "。<span style='color:orange;'>观察提示：关注后续变化趋势。</span>"
        elif level == 'low':
            explanation += "。<span style='color:blue;'>观察提示：继续观察。</span>"
        else:
            explanation += "，在正常波动范围内。"

        # 结合医学参考范围提供额外上下文
        ref_lower, ref_upper = ref_range
        clinical_context = []
        if pd.notna(ref_upper) and current_value > ref_upper:
            # 超出上限多少百分比
            over_pct = (current_value - ref_upper) / (ref_upper + 1e-9) * 100
            clinical_context.append(
                f'已超过医学参考上限(<b>{ref_upper}</b>) <b>{over_pct:.0f}%</b>'
            )
        elif pd.notna(ref_lower) and current_value < ref_lower:
            # 低于下限多少百分比
            under_pct = (ref_lower - current_value) / (ref_lower + 1e-9) * 100
            clinical_context.append(
                f'已低于医学参考下限(<b>{ref_lower}</b>) <b>{under_pct:.0f}%</b>'
            )

        return {
            'level': level,
            'severity': severity,
            'recommendation': recommendation,
            'clinical_context': clinical_context, # 列表形式
            'interpretation': explanation # HTML格式解释
        }
    
def detect_anomalies(
    patient_data: pd.DataFrame,
    weights: pd.Series,
    trained_models: dict,
    current_phase: str
) -> dict:
    """
    【V4.0.4 新增】基础的 IsolationForest 异常检测
    
    此函数运行纯净的模型检测，并（关键地）返回 SHAP 贡献值
    以供上层函数 (boost) 使用。
    """
    logger.debug(f"运行基础异常检测 (IsolationForest) for phase: {current_phase}")
    
    # 1. 获取模型包
    model_pack = trained_models.get(current_phase)
    if not model_pack or 'model' not in model_pack:
        return {'is_anomaly': False, 'message': f'没有可用的模型: {current_phase}'}

    try:
        model = model_pack['model']
        imputation = model_pack.get('imputation', pd.Series(dtype=float))
        background_data = model_pack.get('background_data')
        features_trained = model_pack.get('features_trained', [])
        
        if background_data is None or background_data.empty or not features_trained:
             return {'is_anomaly': False, 'message': f'模型 {current_phase} 数据不完整'}

        # 2. 获取最新数据点
        latest_data_point_raw = patient_data.iloc[-1:]
        
        # 3. 数据对齐与加权
        data_point_aligned = latest_data_point_raw.reindex(
            columns=features_trained
        ).fillna(imputation).fillna(0)
        
        aligned_data, aligned_weights = data_point_aligned.align(
            weights, axis=1, join='inner'
        )
        
        weighted_data_point = aligned_data * aligned_weights
        
        if weighted_data_point.empty:
            return {'is_anomaly': False, 'message': '数据与权重对齐失败'}
        
        # 4. 获取异常分数
        # IsolationForest: 负分是异常, 0 附近是边界, 正分是正常
        score = model.decision_function(weighted_data_point)[0]
        
        # 阈值 (offset_) 通常是负值。分数低于它 = 异常
        threshold = model.offset_ 
        
        is_anomaly = score < threshold
        
        # 5. 计算 SHAP 值 (为 'boost' 函数做准备)
        feature_contributions = {}
        expected_value = 0.0

        if is_anomaly: # 只有当统计上异常时，才计算 SHAP
            logger.debug(f"检测到统计异常 (Score: {score:.3f} < {threshold:.3f})，计算 SHAP...")
            shap_results = explainability.get_shap_explanation(
                model, 
                background_data,
                weighted_data_point,
                compute_interactions=False 
            )
            
            if shap_results and "shap_values_obj" in shap_results:
                shap_obj = shap_results["shap_values_obj"]
                shap_values_flat = shap_obj.values.flatten()
                feature_names = shap_obj.feature_names
                feature_contributions = dict(zip(feature_names, shap_values_flat))
                expected_value = shap_obj.base_values.flatten()[0] # 获取 E[f(x)]
            else:
                logger.warning("SHAP 计算失败，无法进行增强决策")
        else:
             logger.debug(f"统计正常 (Score: {score:.3f} >= {threshold:.3f})，跳过 SHAP")

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'threshold': threshold,
            'feature_contributions': feature_contributions, # 传递特征贡献
            'expected_value': expected_value, # 传递 SHAP 基准值
            'message': f'Score: {score:.3f} vs Threshold: {threshold:.3f}'
        }
    except Exception as e:
        logger.error(f"detect_anomalies 内部失败: {e}", exc_info=True)
        return {'is_anomaly': False, 'message': '检测时发生内部错误'}


def detect_anomalies_with_shap_boost(
    patient_data: pd.DataFrame,
    weights: pd.Series,
    trained_models: dict,
    patient_id: int,
    current_phase: str
) -> dict:
    """
    带 SHAP 增强的异常检测（返回真实归因类型）
    
    核心逻辑：
    1. 基线模型（IsolationForest）使用纯净权重检测统计异常
    2. SHAP 经验用于调整【风险评分】，而非修改模型
    """
    # ========================================
    # 第 1 层：纯净的统计异常检测
    # ========================================
    
    baseline_result = detect_anomalies(
        patient_data, 
        weights, 
        trained_models, 
        current_phase
    )
    
    if not baseline_result['is_anomaly']:
        # 如果基线检测正常，直接返回
        return baseline_result

    # 设置归因类型为 'real'（因为来自模型的真实 SHAP）
    baseline_result['shap_type'] = 'real'

    # ========================================
    # 第 2 层：SHAP 增强决策
    # ========================================
    shap_stats = data_manager.load_feature_importance(patient_id)
    
    if shap_stats.empty:
        # 无 SHAP 经验，返回原始结果
        logger.debug("无 SHAP 经验，返回基线检测结果")
        return baseline_result
    
    # 提取基线结果
    original_score = baseline_result['anomaly_score']
    feature_contributions = baseline_result.get('feature_contributions', {})
    base_score = baseline_result.get('expected_value', 0.0) # E[f(x)]
    
    # 【修正版调整逻辑】
    # 我们重新计算总分：Adjusted Score = Base + SUM(contribution * multiplier)
    adjusted_score = base_score
    adjustment_details = []
    
    for feature, contribution in feature_contributions.items():
        if feature not in shap_stats.index:
            adjusted_score += contribution # 无经验，按原贡献累加
            continue
        
        consistency = shap_stats.loc[feature].get('shap_consistency', 0)
        
        # 默认乘子
        multiplier = 1.0
        
        # 调整逻辑 (基于 V5.0 参数)
        if consistency > 0.7:
            multiplier = 1.3 # 放大高一致性特征
            adjustment_details.append(
                f"**{feature}**: 历史经验显示此指标**高度一致** (Cons={consistency:.1f})，"
                f"风险贡献度 **×{multiplier}**"
            )
        elif abs(consistency) < 0.3:
            multiplier = 0.7 # 抑制方向不一致的特征
            adjustment_details.append(
                f"**{feature}**: 历史经验显示此指标**方向不一致** (Cons={consistency:.1f})，"
                f"风险贡献度 **×{multiplier}**"
            )
        
        # 累加调整后的贡献
        adjusted_score += (contribution * multiplier)
    
    # ========================================
    # 第 3 层：返回增强结果
    # ========================================
    baseline_result['original_score'] = original_score
    baseline_result['adjusted_score'] = adjusted_score
    baseline_result['shap_adjustments'] = adjustment_details
    
    # 使用相同的阈值判断调整后的分数
    baseline_result['is_anomaly'] = adjusted_score < baseline_result['threshold']
    
    logger.info(
        f"SHAP 增强完成: 原始分数={original_score:.3f}, "
        f"调整后={adjusted_score:.3f} (阈值={baseline_result['threshold']:.3f})"
    )
    
    return baseline_result
