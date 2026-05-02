# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

"""
业务逻辑控制器 - 统一管理数据流和模型调用
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict
import config
import app_state
import data_manager
import feature_engineering
import analysis_engine
import risk_engine
import explainability
import shap
import logging
import baseline_monitor
logger = logging.getLogger(__name__)


class HealthController:
    """CareTrace 健康数据分析主控制器"""
    
    def __init__(self, patient_id: int, state=None):
        self.patient_id = patient_id
        self.state = state or app_state.DictStateStore()
        self.cache = {}  # 不再用于存储 MOGP
        self.baseline_monitor = baseline_monitor.PersonalBaselineMonitor()
        self.baseline_monitor_calibrated = False # 标记是否已校准
        self._models = None
        self._weights = None
        self._cleanup_incomplete_mogp_cache()  # 自动检测并清理残缺数据
        logger.info(f"HealthController (PID: {self.patient_id}) 已初始化 (含基线监测器)。")


    @property
    def models(self):
        """懒加载模型属性 (V2 - 健壮版)"""
        if self._models is None:
            logger.debug(f"从磁盘加载模型 (PID: {self.patient_id})...")
            
            # 1. 正常加载
            models_dict, self._weights, _ = data_manager.load_model_and_weights(self.patient_id)
            # 2. 检查加载结果是否为 None（即文件不存在）
            if models_dict is None:
                logger.warning(f"加载模型返回 None (PID: {self.patient_id})，将 self._models 设置为空字典 {{}}")
                self._models = {}  # 关键：返回空字典，而不是 None
            else:
                self._models = models_dict
                
        return self._models
    
    @property
    def weights(self):
        """懒加载权重属性 (V2 - 健壮版)"""
        if self._weights is None and self._models is None:
            models_dict, self._weights, _ = data_manager.load_model_and_weights(self.patient_id)
            
            # 同步处理 models
            if models_dict is None:
                self._models = {}
            else:
                self._models = models_dict
                
        return self._weights  # 注意：权重可以为 None（表示未训练）


    def _cleanup_incomplete_mogp_cache(self):
        """
        【V2.0 - 利用兼容性逻辑】
        使用 data_manager.load_mogp_results 加载数据（内置兼容性处理），
        然后检查修复后的结果是否完整。
        """
        logger.debug("MOGP 缓存健康检查...")
        
        # 1. 使用 load_mogp_results 加载并修复数据
        #    （它内部会自动处理 all_predicted_mean → predicted_mean）
        mogp_results, _, _ = data_manager.load_mogp_results(self.patient_id)
        
        if mogp_results:
            is_incomplete = False
            
            # 2. 检查修复后的数据是否完整
            for indicator, data in mogp_results.items():
                # 【关键】检查必需的键是否存在且有数据
                required_keys = ['all_dates', 'all_predicted_mean', 'all_uncertainty', 'all_trend']
                
                for key in required_keys:
                    if key not in data:
                        logger.warning(
                            f"⚠️ 指标 {indicator} 缺少必需的键 '{key}'，标记为残缺"
                        )
                        is_incomplete = True
                        break
                        
                    # 检查值是否为空
                    if not data[key]:  # 空列表、None、空字符串
                        logger.warning(
                            f"⚠️ 指标 {indicator} 的键 '{key}' 为空，标记为残缺"
                        )
                        is_incomplete = True
                        break
                
                if is_incomplete:
                    break  # 无需检查更多指标
            
            # 3. 如果数据残缺，执行清理
            if is_incomplete:
                logger.warning("检测到残缺的 MOGP 数据，执行清理...")
                try:
                    with data_manager.get_db_connection() as conn:
                        conn.execute(
                            "DELETE FROM mogp_predictions WHERE patient_id = ?",
                            (self.patient_id,)
                        )
                        conn.commit()
                    logger.info("✅ 残缺的 MOGP 数据库记录已清理。")
                    
                    # 清理外部状态
                    self._clear_mogp_state()
                    
                except Exception as e:
                    logger.error(f"❌ 清理失败: {e}", exc_info=True)
            else:
                logger.debug("MOGP 数据完整，无需清理。")
        else:
            logger.debug("未找到 MOGP 数据。")
    
    
    def _clear_mogp_state(self):
        """清理外部状态中的 MOGP 缓存。"""
        self.state.clear_mogp()
        logger.info("✅ 已清空状态存储中的 MOGP 缓存（含处理版本）")


    def _calibrate_baseline_monitor_if_needed(self, force: bool = False):
        """
        如果需要，加载数据并校准基线监测器
        """
        if not force and self.baseline_monitor_calibrated:
            logger.debug("基线监测器已校准，跳过。")
            return

        logger.info("需要校准基线监测器，正在加载数据...")
        # 直接调用 data_manager，避免循环依赖
        try:
            patient_data_raw = data_manager.load_patient_data(self.patient_id)
            if not patient_data_raw.empty:
                self.baseline_monitor.calibrate(patient_data_raw)  # 传入原始数据
                self.baseline_monitor_calibrated = True
                logger.info("基线监测器校准完成。")
            else:
                logger.warning("校准跳过：无患者数据。")
        except Exception as e:
            logger.error(f"校准基线监测器失败: {e}", exc_info=True)
            # 即使校准失败，也标记为尝试过，避免重复失败
            self.baseline_monitor_calibrated = True # 标记为尝试过


    
    def get_processed_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        获取处理后的患者数据 (V3 - 使用可替换状态存储进行版本跟踪)
        
        :param force_refresh: 是否强制刷新缓存
        :return: (原始数据, 特征工程后数据, 参考范围)
        """
        cache_key = f"data_{self.patient_id}"
        
        # 使用状态存储比较 MOGP 版本
        current_mogp_version = self.state.get('mogp_last_updated')
        last_processed_version = self.state.get('processed_mogp_version')
        
        # 比较版本（处理 None 的情况）
        mogp_changed = (current_mogp_version != last_processed_version)
        
        if mogp_changed:
            # 增强日志：区分不同场景
            if current_mogp_version is None and last_processed_version is not None:
                reason = "MOGP 被清空或重置"
            elif current_mogp_version is not None and last_processed_version is None:
                reason = "首次加载 MOGP 数据（可能从数据库恢复）"
            else:
                reason = "MOGP 数据已更新（重新训练）"
            
            logger.info(
                f"检测到 MOGP 版本变更 - 原因: {reason}\n"
                f"  当前版本: {current_mogp_version}\n"
                f"  上次处理: {last_processed_version}\n"
                f"  → 强制刷新数据处理流程"
            )
            force_refresh = True
        
        # --- 缓存命中检查 ---
        if not force_refresh and cache_key in self.cache:
            logger.debug(
                f"get_processed_data (PID: {self.patient_id}): 命中缓存 "
                f"(MOGP版本: {current_mogp_version})"
            )
            return self.cache[cache_key]
        
        logger.info(
            f"get_processed_data (PID: {self.patient_id}): "
            f"{'强制刷新' if force_refresh else '缓存未命中'}，正在处理数据..."
        )
        
        # --- 1. 加载原始数据 ---
        patient_data_raw = data_manager.load_patient_data(self.patient_id)
        
        # --- 2. 加载参考范围 ---
        references_df = data_manager.load_references()
        ref_ranges_dict = {
            index: (row['lower_bound'], row['upper_bound'])
            for index, row in references_df.iterrows()
        }
        
        # --- 3. 特征工程 ---
        if not patient_data_raw.empty:
            patient_data = feature_engineering.add_change_rate_features(
                patient_data_raw, ref_ranges_dict
            )
            
            # --- 4. 合并 MOGP 特征（如果存在）---
            mogp_results = self.state.get('mogp_results')
            if mogp_results:
                logger.debug(f"检测到 MOGP 结果，尝试合并 {len(mogp_results)} 个指标的特征...")
                patient_data = self._merge_mogp_features(patient_data, mogp_results)
            else:
                logger.debug("无 MOGP 结果，跳过特征合并")
        else:
            patient_data = pd.DataFrame()
        
        # --- 5. 缓存结果 ---
        result = (patient_data_raw, patient_data, ref_ranges_dict)
        self.cache[cache_key] = result
        logger.debug(
            f"(PID: {self.patient_id}) 数据已处理并存入缓存。"
            f"Raw: {patient_data_raw.shape}, Processed: {patient_data.shape}"
        )
        
        # 处理完成后，更新"已处理版本"标记
        self.state.set('processed_mogp_version', current_mogp_version)
        logger.debug(f"✅ 更新 'processed_mogp_version' to: {current_mogp_version}")
        
        return result
    
    
    def _merge_mogp_features(self, patient_data: pd.DataFrame, mogp_results: dict) -> pd.DataFrame:
        """
        【辅助方法】将 MOGP 特征合并到患者数据中
        
        :param patient_data: 已完成特征工程的患者数据
        :param mogp_results: MOGP 预测结果字典
        :return: 合并后的数据
        """
        logger.info(f"正在合并 MOGP 特征...")
        
        # 【新增】先清理已存在的 MOGP 特征列，避免重复
        existing_gp_cols = [col for col in patient_data.columns if '_gp_' in col]
        if existing_gp_cols:
            logger.debug(f"  检测到已存在的 MOGP 特征列 ({len(existing_gp_cols)} 个)，先清理...")
            patient_data = patient_data.drop(columns=existing_gp_cols)
        
        temp_dfs_to_join = []
        
        for indicator, data in mogp_results.items():
            required_keys = ['all_dates', 'all_predicted_mean', 'all_uncertainty', 'all_trend']
            if not all(k in data for k in required_keys):
                logger.warning(f"MOGP 数据 {indicator} 不完整，跳过")
                continue
            
            try:
                # 统一移除时区信息
                dates_raw = pd.to_datetime(data['all_dates'])
                
                # 处理多种时区情况
                if dates_raw.tz is not None:
                    dates_normalized = dates_raw.tz_localize(None)
                    logger.debug(f"  {indicator}: 已移除 MOGP 日期的时区 ({dates_raw.tz})")
                else:
                    dates_normalized = dates_raw
                    logger.debug(f"  {indicator}: MOGP 日期本身无时区")
                
                temp_gp_df = pd.DataFrame({
                    'report_date': dates_normalized,
                    f'{indicator}_gp_predicted_mean': np.array(data['all_predicted_mean'], dtype=np.float64),
                    f'{indicator}_gp_uncertainty_std': np.array(data['all_uncertainty'], dtype=np.float64),
                    f'{indicator}_gp_uncertainty_trend': np.array(data['all_trend'], dtype=np.float64)
                }).set_index('report_date')
                
                temp_dfs_to_join.append(temp_gp_df)
                
            except Exception as e:
                logger.error(f"MOGP 特征 {indicator} 转换失败: {e}", exc_info=True)
        
        if temp_dfs_to_join:
            all_gp_features_df = pd.concat(temp_dfs_to_join, axis=1)
            
            # 确保 patient_data 索引也无时区
            patient_data.index = pd.to_datetime(patient_data.index)
            if patient_data.index.tz is not None:
                patient_data.index = patient_data.index.tz_localize(None)
                logger.debug("  已移除 patient_data 索引的时区")
            
            all_gp_features_df.index = pd.to_datetime(all_gp_features_df.index)
            if all_gp_features_df.index.tz is not None:
                all_gp_features_df.index = all_gp_features_df.index.tz_localize(None)
                logger.debug("  已移除 GP features 索引的时区")
            
            # 执行 join 前，最后验证
            logger.debug(
                f"  准备 join: patient_data.index.tz={patient_data.index.tz}, "
                f"gp_features.index.tz={all_gp_features_df.index.tz}"
            )
            
            patient_data = patient_data.join(all_gp_features_df, how='left')
            logger.info(f"✅ 成功合并 {len(temp_dfs_to_join)} 个指标的 MOGP 特征")
        else:
            logger.warning("⚠️ 没有可合并的 MOGP 特征")
        
        return patient_data
    
    def _train_mogp_models(self, surveillance_df_raw: pd.DataFrame) -> dict:
        """
        【V4.0.4 - 阶段纯净版】独立训练 MOGP 模型（每个指标单独训练）
        
        核心改进：
        1. 在传递给 select_mogp_indicators 之前，过滤掉所有衍生特征
        2. 确保 MOGP 只看"原始化验值"，不受跨期特征污染
        """
        logger.info(f"(PID: {self.patient_id}) MOGP: 正在选择指标（V4.0.4 - 纯净版）...")
        
        # 提取原始指标列表
        original_indicators = [
            col for col in surveillance_df_raw.columns
            if not config.DERIVED_FEATURE_PATTERN.search(col)  # 排除 _rate, _accel, _ewma 等
            and '_transition_shock' not in col                # 显式排除过渡特征
            and '_gp_' not in col                           # 排除旧的 MOGP 特征（如果存在）
            and col not in ['report_uuid', 'phase', 'user_label', 'id', 'patient_id']
        ]
        
        logger.debug(
            f"  原始 DataFrame 列数: {len(surveillance_df_raw.columns)}\n"
            f"  过滤后原始指标数: {len(original_indicators)}\n"
            f"  示例: {original_indicators[:5]}"
        )
        
        if not original_indicators:
            logger.error("❌ MOGP: 过滤后无可用指标（可能全是衍生特征）")
            return {
                'success': False,
                'message': '无原始化验指标可用于 MOGP 训练',
                'diagnostic_info': {}
            }
        
        # 只传递原始指标给选择器
        surveillance_df_clean = surveillance_df_raw[original_indicators].copy()
        
        # 使用 risk_engine 模块的函数（现在输入是纯净的）
        default_indicators, diagnostic_info = risk_engine.select_mogp_indicators(
            surveillance_df_clean,  # 只包含原始化验指标
            max_indicators=4
        )
        
        if not default_indicators:
            return {
                'success': False,
                'message': '暂无符合条件的指标（需>=5个数据点）',
                'diagnostic_info': diagnostic_info
            }
        
        logger.info(
            f"(PID: {self.patient_id}) MOGP: 选定 {len(default_indicators)} 个指标进行训练:\n"
            f"  {default_indicators}"
        )
    
        # 加载参考范围
        try:
            references_df = data_manager.load_references()
            ref_ranges_dict = {
                index: (row['lower_bound'], row['upper_bound'])
                for index, row in references_df.iterrows()
            }
            logger.debug(f"(PID: {self.patient_id}) MOGP: 成功加载 {len(ref_ranges_dict)} 个指标的参考范围")
        except Exception as e:
            logger.warning(f"(PID: {self.patient_id}) MOGP: 加载参考范围失败: {e}，将使用默认值")
            ref_ranges_dict = {}
    
        # 逐个指标训练（使用纯净数据）
        all_results = {}
        successful_indicators = []
        
        for indicator in default_indicators:
            # 确保从纯净数据中提取
            single_indicator_data = surveillance_df_clean[[indicator]].dropna()
            
            # --- 数据量和突变预检查）---
            
            # 检查1：数据点是否足够训练 GP
            if len(single_indicator_data) < config.MIN_GP_POINTS:
                logger.warning(
                    f"(PID: {self.patient_id}) MOGP: ⚠️ 跳过 {indicator} - "
                    f"数据点不足 ({len(single_indicator_data)} < {config.MIN_GP_POINTS})。"
                    "无法生成可靠趋势预测。"
                )
                continue  # 跳过此指标
            
            # 检查2：如果数据点较少，检查是否有突变
            if len(single_indicator_data) < config.MIN_POINTS_FOR_SPIKE_PREDICT:
                logger.debug(
                    f"(PID: {self.patient_id}) MOGP: {indicator} 数据点 ({len(single_indicator_data)}) "
                    "较少，执行突变检测（带临床上下文）..."
                )
                
                try:
                    # 获取参考范围
                    ref_lower, ref_upper = ref_ranges_dict.get(indicator, (None, None))
                    
                    # 构建临床上下文
                    clinical_context = {
                        'phase': '稳定监控期',
                        'baseline_value': single_indicator_data[indicator].iloc[0] if len(single_indicator_data) > 0 else None,
                        'upper_limit': ref_upper
                    }
                    
                    values = single_indicator_data[indicator].values
                    trend = analysis_engine.detect_trend_change(values, context=clinical_context)
                    
                    if trend['has_spike']:
                        logger.warning(
                            f"(PID: {self.patient_id}) MOGP: ⚠️ 跳过 {indicator} - "
                            f"数据点不足 ({len(single_indicator_data)} < {config.MIN_POINTS_FOR_SPIKE_PREDICT}) "
                            f"且检测到显著突变 (变化率: {trend['spike_ratio']:.1%}, 阈值: {trend['threshold']:.1%}, "
                            f"上下文调整: {trend.get('context_adjusted', False)})。"
                            "预测可能不可靠，请等待更多数据后再观察。"
                        )
                        continue  # 跳过此指标
                    else:
                        logger.debug(f"(PID: {self.patient_id}) MOGP: {indicator} 未检测到显著突变，继续训练。")
                
                except Exception as detect_e:
                    logger.warning(
                        f"(PID: {self.patient_id}) MOGP: {indicator} 突变检测失败: {detect_e}",
                        exc_info=True
                    )
                    # 选择继续尝试训练（保守策略）
                    pass
            
            # 数据充足且（如果检查了）无显著突变，则尝试训练
            logger.debug(f"(PID: {self.patient_id}) MOGP: 正在训练 {indicator} ({len(single_indicator_data)} 个点)...")
            try:
                # ✅ 传递纯净的单指标数据
                result = analysis_engine.train_and_predict_mogp(
                    single_indicator_data,  # 仅包含原始指标值，无衍生特征
                    [indicator]
                )
                if result and indicator in result:
                    all_results[indicator] = result[indicator]
                    successful_indicators.append(indicator)
                    logger.debug(f"(PID: {self.patient_id}) MOGP: ✅ {indicator} 训练成功。")
                else:
                    logger.warning(f"(PID: {self.patient_id}) MOGP: ⚠️ {indicator} 训练返回了空结果。")
            except Exception as e:
                logger.error(f"(PID: {self.patient_id}) MOGP: ❌ {indicator} 训练失败: {e}", exc_info=True)
        
        # 保存结果
        if all_results:
            logger.info(f"(PID: {self.patient_id}) MOGP: 共 {len(successful_indicators)} 个指标训练成功，正在保存到数据库...")
            # 持久化到数据库
            data_manager.save_mogp_results(
                self.patient_id, all_results, successful_indicators
            )
            
            return {
                'success': True,
                'indicators': successful_indicators,
                'results': all_results,
                'diagnostic_info': diagnostic_info
            }
        
        logger.warning(f"(PID: {self.patient_id}) MOGP: 所有指标均训练失败。")
        return {
            'success': False,
            'message': '所有指标训练失败',
            'diagnostic_info': diagnostic_info
        }


    def train_models(self) -> dict:
        """
        训练所有模型（IsolationForest + MOGP）
        
        核心改进：
        1. 避免重复特征工程（内存合并 MOGP）
        2. 修复解包错误
        3. 修复条件保存逻辑
        4. 完整实现 SHAP 学习闭环
        """
        logger.info(f"===== 开始训练模型 (PID: {self.patient_id}) =====")
        
        # --- 1. 获取基础特征工程后的数据 ---
        patient_data_raw, patient_data_base, ref_ranges_dict = self.get_processed_data()
        
        if patient_data_base.empty:
            logger.warning(f"训练失败：处理后数据为空")
            return {'success': False, 'message': '处理后数据为空'}
        
        # --- 2. 提取稳定期原始数据用于 MOGP ---
        surveillance_df_raw = patient_data_raw[
            patient_data_raw['phase'] == '稳定监控期'
        ].copy()
        
        # --- 3. 训练 MOGP 模型 ---
        logger.info("准备训练 MOGP 模型...")
        mogp_result = self._train_mogp_models(surveillance_df_raw)
        
        # --- 4. 在内存中合并 MOGP 特征 ---
        if mogp_result.get('success'):
            try:
                mogp_results_dict = mogp_result.get('results')
                patient_data_full = self._merge_mogp_features(
                    patient_data_base,
                    mogp_results_dict
                )
                logger.info("✅ MOGP 特征已合并到训练数据")
                
                # 立即更新版本追踪器
                current_mogp_version = self.state.get('mogp_last_updated')
                self.state.set('processed_mogp_version', current_mogp_version)
                logger.debug(f"✅ 更新 'processed_mogp_version' to: {current_mogp_version}")
                
                # 同步到外部状态
                self.state.update({
                    'mogp_results': mogp_results_dict,
                    'mogp_target_indicators': mogp_result.get('indicators'),
                    'mogp_last_updated': datetime.now(),
                    'mogp_diagnostic_info': mogp_result.get('diagnostic_info')
                })
                
            except Exception as e:
                logger.error(f"❌ MOGP 特征合并失败: {e}", exc_info=True)
                patient_data_full = patient_data_base  # 回退
        else:
            logger.warning("MOGP 训练失败，使用基础特征继续")
            patient_data_full = patient_data_base
            self.state.set('mogp_results', None)
        
        # --- 5. 提取特征列（基于最终数据）---
        numeric_cols = patient_data_full.select_dtypes(include='number').columns

        all_feature_columns = [
            col for col in numeric_cols 
            if col not in ['report_uuid', 'id', 'patient_id']
            and '_transition_shock' not in col
        ]
        
        if not all_feature_columns:
            logger.error("训练失败：无数值特征")
            return {'success': False, 'message': '无数值特征可训练'}

        logger.debug(
            f"✅ 特征工程完成：\n"
            f"  - 原始特征数: {len(numeric_cols)}\n"
            f"  - 过滤后: {len(all_feature_columns)}\n"
            f"  - 已排除: _transition_shock（跨界污染特征）"
        )

        # --- 6. 训练 IsolationForest ---
        logger.info(f"准备训练 IsolationForest ({len(all_feature_columns)} 个特征)...")
        
        # 正确解包：train_complete_pipeline 返回 (models_dict, final_weights)
        models_dict, weights = analysis_engine.train_complete_pipeline(
            patient_data_full,
            all_feature_columns,
            self.patient_id
        )
        
        if models_dict is None or weights is None:
            message = f'IsolationForest 训练失败 (需至少 {config.MIN_SAMPLES_FOR_MODEL} 条数据)'
            return {
                'success': True,
                'isolation_forest': False,
                'message': message,
                'mogp': mogp_result
            }
        
        # --- 7. 条件保存模型和权重 ---
        fallback_activated = models_dict.get('fallback_activated', False)
        single_phase_mode = models_dict.get('single_phase_mode', False)
        
        if fallback_activated:
            # 【场景1】权重计算完全失败（无双阶段数据）
            logger.warning(
                "⚠️ 权重计算失败（缺少'治疗期'或'稳定期'数据）\n"
                "  → 跳过权重保存，避免污染历史数据\n"
                "  → SHAP 学习闭环也将被跳过"
            )
            data_manager.save_model_and_weights(
                self.patient_id, 
                models_dict,
                weights=None,
                feature_columns=all_feature_columns
            )
            self._models = models_dict
            self._weights = None
            
            # 更新缓存
            self.cache[f"data_{self.patient_id}"] = (
                patient_data_raw, patient_data_full, ref_ranges_dict
            )
            
            # 立即返回，避免后续流程使用无效权重
            logger.info(f"===== 模型训练结束 (PID: {self.patient_id}) [回退模式] =====")
            return {
                'success': True,
                'isolation_forest': True,
                'mogp': mogp_result,
                'fallback_activated': True,
                'single_phase_mode': False,
                'message': '⚠️ 权重计算失败，需补充数据后重新训练',
                'boost_df': None,
                'dynamic_params': models_dict.get('dynamic_params')
            }
        
        elif single_phase_mode:
            # 【场景2】单阶段模式（基于 CV 的权重）
            logger.warning(
                "⚠️ 使用单阶段权重（基于 CV）\n"
                "  → 权重可靠性较低，将保存但标记为 'provisional'"
            )
            data_manager.save_model_and_weights(
                self.patient_id, 
                models_dict, 
                weights, 
                feature_columns=all_feature_columns
            )
            self._models = models_dict
            self._weights = weights
            
            status_message = '✅ 模型训练完成（⚠️ 使用单阶段权重）'
            
        else:
            # 【场景3】正常双阶段（个性化权重）
            logger.info("✅ 使用双阶段个性化权重，正在保存...")
            data_manager.save_model_and_weights(
                self.patient_id, 
                models_dict, 
                weights, 
                feature_columns=all_feature_columns
            )
            self._models = models_dict
            self._weights = weights
            
            status_message = '✅ 模型训练完成（使用个性化权重）'
        
        # --- 8. 更新缓存 ---
        self.cache[f"data_{self.patient_id}"] = (
            patient_data_raw, patient_data_full, ref_ranges_dict
        )
        logger.info("✅ 缓存已更新")
        
        # --- 9. 返回结果 ---
        logger.info(f"===== 模型训练结束 (PID: {self.patient_id}) =====")
        return {
            'success': True,
            'isolation_forest': True,
            'mogp': mogp_result,
            'fallback_activated': False,
            'single_phase_mode': single_phase_mode,
            'boost_df': models_dict.get('boost_df'),
            'dynamic_params': models_dict.get('dynamic_params'),
            'message': status_message
        }


    def assess_current_risk(self) -> dict:
        """
        观察当前健康数据模式 - 融合基线偏离检测 + 确保所有警报都有 SHAP 归因
        
        核心改进：
        1. 为 Z-Score 警报添加代理 SHAP（100% 归因给检测指标）
        2. 确保模型警报已有真实 SHAP（已在 baseline_monitor 中实现）
        3. 为启发式警报添加代理 SHAP
        """
        cache_key = f"risk_assessment_{self.patient_id}"

        # 检查标签刷新标志
        if self.state.get('labels_changed', False):
            logger.info("检测到用户反馈标签变更，清除风险评估缓存")
            self.cache.pop(cache_key, None)
            self.state.set('labels_changed', False)

        # 带时间戳的短期缓存
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                logger.info("使用缓存的风险评估结果（5分钟内）")
                return cached_data
        
        logger.info(f"开始评估当前风险 (PID: {self.patient_id}) (V7.1 版本)...")

        # --- 0. 校准 Z-Score 监测器 ---
        try:
            self._calibrate_baseline_monitor_if_needed()
        except Exception as cal_e:
            logger.error(f"评估前校准基线监测器失败: {cal_e}", exc_info=True)

        # --- 1. 加载数据 ---
        patient_data_raw, patient_data, ref_ranges_dict = self.get_processed_data()

        if patient_data.empty or len(patient_data) < 1:
            logger.warning(f"(PID: {self.patient_id}) 风险评估失败：无数据可分析。")
            result = {
                'attention_level': 'unknown',
                'message': '无数据可分析',
                'observations': [],
                'summary_statement': '无数据',
                'disclaimer': risk_engine.DISCLAIMER
            }
            return result

        # --- 2. 获取最新数据点信息 ---
        latest_data = patient_data.iloc[-1]
        current_phase = latest_data.get('phase', '未知')
        current_uuid = latest_data.get('report_uuid', 'N/A')
        logger.debug(
            f"(PID: {self.patient_id}) 使用日期 {latest_data.name} "
            f"(Phase: {current_phase}) 的数据进行评估。"
        )

        # ========================================
        # 警报源 1：统计基线偏离 (Z-Score)
        # ========================================
        baseline_alerts = []
        tumor_markers = [
            item['name'] for item in config.LAB_REPORT_CONFIG.get('肿瘤标志物', [])
        ]
        logger.debug(
            f"准备对 {len(tumor_markers)} 个肿瘤标志物进行 "
            f"[统计基线 Z-Score] 检测..."
        )

        for marker in tumor_markers:
            if marker in patient_data.columns:
                try:
                    baseline_result = self.baseline_monitor.assess_current_value(
                        patient_data, marker, ref_ranges_dict
                    )

                    if (baseline_result.get('status') == 'success' and 
                        baseline_result.get('level') != 'normal'):
                        
                        # 为 Z-Score 警报添加代理 SHAP
                        alert_obs = {
                            'id': f'BASELINE_DEVIATION_{baseline_result["level"].upper()}_{marker.replace(" ", "_")}',
                            'indicator': marker,
                            'report_uuid': current_uuid,
                            'pattern_type': 'z_score',
                            'observation': baseline_result.get('interpretation', ''),
                            'data_context': (
                                f"当前: {baseline_result.get('current_value', np.nan):.2f}, "
                                f"基线: {baseline_result.get('baseline', {}).get('median', np.nan):.2f} "
                                f"(Z={baseline_result.get('deviation', {}).get('modified_z_score', np.nan):.1f}σ)"
                            ),
                            'note': baseline_result.get('recommendation', ''),
                            'attention_level': baseline_result.get('level', 'low'),
                            'score_weight': {
                                'high': 60, 
                                'medium': 35, 
                                'low': 15
                            }.get(baseline_result.get('level'), 0),
                            'severity': baseline_result.get('severity', ''),
                            'current_value': baseline_result.get('current_value'),
                            'baseline_median': baseline_result.get('baseline', {}).get('median'),
                            'z_score': baseline_result.get('deviation', {}).get('modified_z_score'),
                            'baseline_info': baseline_result.get('baseline', {}),
                            
                            # 代理 SHAP（100% 归因给当前指标）
                            'shap_values': {marker: 1.0}
                        }
                        
                        baseline_alerts.append(alert_obs)
                        
                        logger.info(
                            f"  检测到 [统计基线] 偏离: {marker}, "
                            f"Level: {alert_obs['attention_level']}, "
                            f"Z={alert_obs['z_score']:.1f} "
                            f"(已附加代理 SHAP)"
                        )

                except Exception as base_e:
                    logger.error(
                        f"  评估 {marker} 统计基线偏离时出错: {base_e}", 
                        exc_info=True
                    )

        logger.info(
            f"[统计基线] 检测完成，发现 {len(baseline_alerts)} 个警报 "
            f"(全部已附加代理 SHAP)。"
        )
        
        # ========================================
        # 警报源 2：模型基线偏离 (IsolationForest + SHAP Boost)
        # ========================================
        model_anomaly_alerts = []
        logger.debug(
            f"准备进行 [模型基线 SHAP 增强] 检测 (Phase: {current_phase})..."
        )
        
        trained_models = self.models
        trained_weights = self.weights
        
        if trained_models and trained_weights is not None:
            try:
                # 调用 SHAP 增强函数（已在 baseline_monitor 中实现）
                model_result = baseline_monitor.detect_anomalies_with_shap_boost(
                    patient_data,
                    trained_weights,
                    trained_models,
                    self.patient_id,
                    current_phase
                )
                
                if model_result.get('is_anomaly'):
                    logger.info("  检测到 [模型基线] 异常，正在构建警报...")
                    
                    # 格式化 SHAP 调整详情
                    adjust_notes = model_result.get('shap_adjustments', [])
                    if adjust_notes:
                        note = "AI 已根据历史经验调整评分:\n- " + "\n- ".join(adjust_notes)
                    else:
                        note = "基于统计模型的综合判断"
                    
                    # 确保模型警报有 SHAP 值
                    shap_contributions = model_result.get('shap_contributions', {})
                    if not shap_contributions:
                        logger.warning(
                            "⚠️ 模型警报缺少 SHAP 值（不符合预期），"
                            "使用默认归因"
                        )
                        shap_contributions = {'综合健康模型': 1.0}

                    model_alert_obs = {
                        'id': 'MODEL_ANOMALY',
                        'indicator': '综合健康模型',
                        'report_uuid': current_uuid,
                        'pattern_type': 'model_anomaly',
                        'observation': f'综合健康模型检测到异常 (Phase: {current_phase})',
                        'data_context': (
                            f"原始评分: {model_result.get('original_score', 'N/A'):.3f}, "
                            f"调整后: {model_result.get('adjusted_score', 'N/A'):.3f} "
                            f"(阈值: {model_result.get('threshold', 'N/A'):.3f})"
                        ),
                        'note': note,
                        'attention_level': 'high',
                        'score_weight': 70,
                        'model_result': model_result,
                        # 附加真实 SHAP 值（来自模型）
                        'shap_values': shap_contributions
                    }
                    model_anomaly_alerts.append(model_alert_obs)
                    
                    logger.info(
                        f"  ✅ 模型警报已构建（含 {len(shap_contributions)} 个特征的 SHAP 值）"
                    )
                    
            except Exception as model_e:
                logger.error(
                    f"  评估 [模型基线] 偏离时出错: {model_e}", 
                    exc_info=True
                )
        else:
            logger.warning("  跳过 [模型基线] 检测：模型或权重未训练/加载。")
            
        logger.info(
            f"[模型基线] 检测完成，发现 {len(model_anomaly_alerts)} 个警报。"
        )

        # ========================================
        # 警报源 3：启发式规则 (Risk Engine)
        # ========================================
        logger.debug("开始执行 [启发式规则 Risk Engine] 的其他模式观察...")
        
        context = {
            'all_labels': data_manager.get_all_labels(self.patient_id),
            'current_phase_tag': current_phase,
            'current_report_uuid': current_uuid,
            'patient_id': self.patient_id
        }
        
        try:
            pattern_observation_result = risk_engine.observe_health_data_patterns(
                latest_data=latest_data,
                historical_data=patient_data,
                ref_ranges=ref_ranges_dict,
                context=context
            )
            heuristic_alerts = pattern_observation_result.get('observations', [])
            
            # 为启发式警报添加代理 SHAP
            for alert in heuristic_alerts:
                if 'shap_values' not in alert or not alert['shap_values']:
                    # 提取指标名称（优先使用 'indicator'，回退到 'id'）
                    indicator = alert.get('indicator', alert.get('id', 'Unknown'))
                    
                    # 附加代理 SHAP（100% 归因给当前指标）
                    alert['shap_values'] = {indicator: 1.0}
                    
                    # 标记类型（如果尚未标记）
                    if 'pattern_type' not in alert:
                        alert['pattern_type'] = 'heuristic'
                    
                    logger.debug(
                        f"  为启发式警报 '{alert.get('id')}' 添加代理 SHAP: "
                        f"{indicator} = 1.0"
                    )
            
            logger.info(
                f"[启发式规则] 检测完成，发现 {len(heuristic_alerts)} 个警报 "
                f"(已附加代理 SHAP)"
            )
            
        except Exception as risk_e:
            logger.error(
                f"调用 risk_engine.observe_health_data_patterns 失败: {risk_e}", 
                exc_info=True
            )
            heuristic_alerts = []

        # ========================================
        # 6. 融合所有警报
        # ========================================
        logger.debug("开始融合所有警报源...")
        final_observations = baseline_alerts + model_anomaly_alerts + heuristic_alerts
        
        # 确保所有警报都有 SHAP 值
        missing_shap_count = 0
        for obs in final_observations:
            if 'shap_values' not in obs or not obs['shap_values']:
                missing_shap_count += 1
                logger.warning(
                    f"⚠️ 警报 '{obs.get('id')}' 缺少 SHAP 值（不应发生），"
                    f"补充默认值"
                )
                obs['shap_values'] = {obs.get('indicator', 'Unknown'): 1.0}
        
        if missing_shap_count > 0:
            logger.warning(
                f"⚠️ 共 {missing_shap_count}/{len(final_observations)} 个警报缺少 SHAP 值"
            )
        else:
            logger.info(
                f"✅ 所有 {len(final_observations)} 个警报均已附加 SHAP 归因"
            )

        # ========================================
        # 7. 重新排序和计算总分
        # ========================================
        final_observations.sort(
            key=lambda x: x.get('score_weight', 0), 
            reverse=True
        )

        attention_level, level_desc, attention_score, color = (
            risk_engine._calculate_attention_score(final_observations)
        )
        summary = risk_engine._generate_summary_statement(
            attention_level, 
            final_observations
        )

        # ========================================
        # 8. 构建最终结果
        # ========================================
        final_result = {
            'attention_level': attention_level,
            'level_description': level_desc,
            'attention_score': attention_score,
            'color': color,
            'summary_statement': summary,
            'observations': final_observations,
            'observation_count': len(final_observations),
            'high_attention_count': len([
                o for o in final_observations 
                if o.get('attention_level') == 'high'
            ]),
            'medium_attention_count': len([
                o for o in final_observations 
                if o.get('attention_level') == 'medium'
            ]),
            'disclaimer': risk_engine.DISCLAIMER,
            'context': context
        }

        logger.info(
            f"(PID: {self.patient_id}) 风险评估完成。"
            f"最终 Attention Level: {final_result.get('attention_level', 'unknown')}"
        )
        
        self.cache[cache_key] = (final_result, datetime.now())
        return final_result
    
    
    def get_mogp_predictions(self) -> Optional[Dict]:
        """
        从外部状态优先读取，数据库作为后备
        """
        # 1. 优先从外部状态读取
        mogp_results = self.state.get('mogp_results')
        if mogp_results:
            logger.debug(f"从状态存储获取 MOGP 结果")
            return {
                'results': mogp_results,
                'indicators': self.state.get('mogp_target_indicators'),
                'last_updated': self.state.get('mogp_last_updated')
            }
        
        # 2. 外部状态为空，从数据库加载并同步到状态存储
        logger.info(f"状态存储无 MOGP，尝试从数据库加载...")
        mogp_results, indicators, last_updated = data_manager.load_mogp_results(
            self.patient_id
        )
        
        if mogp_results:
            logger.info("从数据库恢复 MOGP 结果并同步到状态存储")
            self.state.update({
                'mogp_results': mogp_results,
                'mogp_target_indicators': indicators,
                'mogp_last_updated': last_updated
            })
            return {
                'results': mogp_results,
                'indicators': indicators,
                'last_updated': last_updated
            }
        
        logger.debug("数据库也无 MOGP 结果")
        return None
    
    def clear_cache(self):
        """清除缓存（包括外部状态中的 MOGP 追踪）"""
        logger.info(f"清除 HealthController 内部缓存 (PID: {self.patient_id})")
        self.cache.clear()

        self._models = None  # 清除模型缓存
        self._weights = None # 清除权重缓存    
            
        # 清除外部状态中的所有 MOGP 相关数据
        self.state.clear_mogp()
        logger.info("已清除状态存储中的 MOGP 缓存及处理版本跟踪器")


    # ========================================
    # 适应性治疗模拟支持
    # ========================================
    
    def get_simulation_context(self) -> Dict:
        """
        为适应性治疗模拟器提供患者数据上下文
        
        :return: {
            'patient_data_raw': 原始化验数据,
            'ref_ranges': 参考范围字典,
            'tumor_markers': 可用的肿瘤标志物列表,
            'latest_values': 最新的标志物值
        }
        """
        logger.info(f"准备患者 {self.patient_id} 的模拟上下文...")
        
        try:
            # 获取原始数据（不需要特征工程）
            patient_data_raw, _, ref_ranges_dict = self.get_processed_data()
            
            if patient_data_raw.empty:
                return {
                    'success': False,
                    'message': '无患者数据'
                }
            
            # 提取肿瘤标志物列表
            tumor_markers = [
                item['name'] for item in config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
            ]
            
            # 过滤出当前患者有数据的标志物
            available_markers = [
                m for m in tumor_markers 
                if m in patient_data_raw.columns and patient_data_raw[m].notna().any()
            ]
            
            if not available_markers:
                return {
                    'success': False,
                    'message': '未找到肿瘤标志物数据'
                }
            
            # 提取最新值
            latest_values = {}
            for marker in available_markers:
                latest_val = patient_data_raw[marker].dropna().iloc[-1]
                latest_values[marker] = float(latest_val)
            
            logger.info(
                f"模拟上下文准备完成：{len(available_markers)} 个标志物，"
                f"最新值: {latest_values}"
            )
            
            return {
                'success': True,
                'patient_data_raw': patient_data_raw,
                'ref_ranges': ref_ranges_dict,
                'tumor_markers': available_markers,
                'latest_values': latest_values
            }
            
        except Exception as e:
            logger.error(f"准备模拟上下文失败: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'数据加载失败: {str(e)}'
            }
