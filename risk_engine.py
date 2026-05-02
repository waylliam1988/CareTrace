# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

"""
CareTrace 关照轨迹 - 个人健康档案的趋势观察工具

⚠️ 重要免责声明:
本系统仅用于个人健康数据的记录和可视化，不提供任何医疗建议、诊断或治疗方案。
所有输出仅为数据统计观察，不能替代专业医生的判断。
任何关于健康状态的决策，请务必咨询有资质的医疗专业人员。
"""

import pandas as pd
import numpy as np
import uuid as uuid_module
from typing import Dict, List
import config
import analysis_engine
import logging
import data_manager
import predictive_ensemble_engine
logger = logging.getLogger(__name__)

# ========================================
# 全局免责声明配置
# ========================================

DISCLAIMER = """
⚠️ **重要提示**: 本系统不提供医疗建议
- 所有分析结果仅为数据统计观察
- 不能替代医生的专业判断
- 任何健康决策请咨询医疗机构
"""


# ========================================
# 1. 指标选择器
# ========================================

def select_mogp_indicators(raw_df: pd.DataFrame, max_indicators: int = 4) -> tuple:
    """
    为趋势追踪智能选择指标（动态化版本）
    
    选择策略（按优先级）:
    1. 肿瘤标志物（从 config 自动读取）
    2. 临床综合指标（NLR, PLR, SII 等）
    3. 血常规核心指标
    4. 生化常规指标
    
    :param raw_df: 原始数据（稳定监控期）
    :param max_indicators: 最多选择的指标数量
    :return: (选中的指标列表, 诊断信息字典)
    """
    logger.info(f"开始为 MOGP 智能选择指标... (max_indicators={max_indicators})")
    selected = []
    candidates = {}
    available_cols = raw_df.columns.tolist()
    
    def check_indicator(indicator_name: str, category: str, priority: int = 1) -> bool:
        """
        检查指标是否可用，并记录诊断信息
        
        :param indicator_name: 指标名称
        :param category: 指标类别（用于诊断）
        :param priority: 优先级（数字越小越优先）
        :return: 是否成功添加到选择列表
        """
        if indicator_name in available_cols:
            count = raw_df[indicator_name].notna().sum()
            sufficient = count >= 5
            logger.debug(f"  MOGP check: {indicator_name} (Cat: {category}, Prio: {priority}) - Found {count} data points. Sufficient: {sufficient}")
            candidates[indicator_name] = {
                'count': count,
                'category': category,
                'priority': priority,
                'sufficient': sufficient
            }
            
            # 只有数据充足且未选择时才添加
            if sufficient and indicator_name not in selected:
                selected.append(indicator_name)
                logger.debug(f"    -> {indicator_name} 已添加到 MOGP 目标列表。")
                return True
        else:
            logger.debug(f"  MOGP check: {indicator_name} (Cat: {category}) - Not found in available columns.")
        return False
    
    # ========================================
    # 优先级 1: 肿瘤标志物（动态读取）
    # ========================================
    logger.debug("MOGP select: 检查 Prio 1 (肿瘤标志物)...")
    tumor_marker_configs = config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
    
    for marker_config in tumor_marker_configs:
        marker_name = marker_config['name']
        
        if check_indicator(marker_name, "肿瘤标志物", priority=1):
            if len(selected) >= max_indicators:
                break
    

    # ========================================
    # 优先级 2: 临床综合指标（衍生特征）
    # ========================================
    if len(selected) < max_indicators:
        logger.debug("MOGP select: 检查 Prio 2 (临床综合指标)...")
        # 从 config 动态读取
        composite_indicators = config.COMPOSITE_INDICATORS
        
        for indicator in composite_indicators:
            if check_indicator(indicator, "临床综合指标", priority=2):
                if len(selected) >= max_indicators:
                    break
    
    # ========================================
    # 优先级 3: 血常规核心指标（动态读取）
    # ========================================
    if len(selected) < max_indicators:
        logger.debug("MOGP select: 检查 Prio 3 (血常规核心指标)...")
        blood_routine_configs = config.LAB_REPORT_CONFIG.get("血常规", [])
        
        # 核心指标的优选顺序
        priority_order = [
            '白细胞计数',
            '中性粒细胞计数',
            '淋巴细胞计数',
            '血红蛋白',
            '血小板计数'
        ]
        
        for indicator_name in priority_order:
            # 检查该指标是否在配置中
            if any(item['name'] == indicator_name for item in blood_routine_configs):
                if check_indicator(indicator_name, "血常规", priority=3):
                    if len(selected) >= max_indicators:
                        break
    
    # ========================================
    # 优先级 4: 生化指标（肝肾功能等）
    # ========================================
    if len(selected) < max_indicators:
        logger.debug("MOGP select: 检查 Prio 4 (生化指标)...")
        biochemistry_configs = config.LAB_REPORT_CONFIG.get("肝肾功能与电解质", [])
        
        # 核心生化指标
        priority_order = [
            '白蛋白 ALB',
            '总蛋白 TP',
            '谷丙转氨酶 ALT',
            '谷草转氨酶 AST',
            '肌酐 Cr'
        ]
        
        for indicator_name in priority_order:
            if any(item['name'] == indicator_name for item in biochemistry_configs):
                if check_indicator(indicator_name, "生化指标", priority=4):
                    if len(selected) >= max_indicators:
                        break
    
    # ========================================
    # 构建诊断信息
    # ========================================
    diagnostic_info = {
        'total_records': raw_df.shape[0],
        'candidates': candidates,
        'selected': selected,
        'selection_summary': {
            '肿瘤标志物': len([s for s in selected if candidates.get(s, {}).get('category') == '肿瘤标志物']),
            '临床综合指标': len([s for s in selected if candidates.get(s, {}).get('category') == '临床综合指标']),
            '血常规': len([s for s in selected if candidates.get(s, {}).get('category') == '血常规']),
            '生化指标': len([s for s in selected if candidates.get(s, {}).get('category') == '生化指标']),
        }
    }

    logger.info(f"MOGP 指标选择完成。共选中 {len(selected)} 个: {selected}")
    logger.debug(f"MOGP 诊断信息: {diagnostic_info}")

    return selected, diagnostic_info


def _calculate_dynamic_drop_threshold(
    marker_series: pd.Series,
    context_series: pd.Series,
    min_samples: int = 3,
    std_multiplier: float = 1.5) -> float | None:
    """
    动态计算治疗期“快速下降”的百分比阈值。
    """
    logger.debug(f"开始为指标 {marker_series.name} 计算动态快速下降阈值...")

    # --- Adaptación: 使用 'phase' 列判断治疗期 ---
    is_treatment_period = context_series == '强效治疗期'
    is_dropping = marker_series.diff() < 0
    historical_drops = marker_series[is_treatment_period & is_dropping]
    logger.debug(f"  找到 {len(historical_drops)} 个治疗期下降数据点。")

    if len(historical_drops) < min_samples:
        logger.warning(f"  数据点不足 ({len(historical_drops)} < {min_samples})，无法计算动态阈值，返回 None。")
        return None

    previous_values = marker_series.shift(1)[historical_drops.index]
    valid_mask = previous_values > 0
    if not valid_mask.any():
        logger.warning("  所有历史下降点的前一个值都<=0，无法计算百分比，返回 None。")
        return None

    drop_percentages = (previous_values[valid_mask] - historical_drops[valid_mask]) / previous_values[valid_mask]

    if len(drop_percentages) < min_samples:
         logger.warning(f"  有效下降百分比数据点不足 ({len(drop_percentages)} < {min_samples})，无法计算动态阈值，返回 None。")
         return None

    mean_drop = drop_percentages.mean()
    std_drop = drop_percentages.std()
    dynamic_threshold = mean_drop + std_multiplier * std_drop
    clipped_threshold = np.clip(dynamic_threshold, 0.15, 0.80) # 安全护栏

    logger.debug(f"  平均下降率={mean_drop:.3f}, 标准差={std_drop:.3f}, 动态阈值={clipped_threshold:.3f}")
    return clipped_threshold


def _create_treatment_obs(
    marker_name: str,
    report_uuid: str,  # ✅ 明确参数含义：真实的化验单UUID
    pattern_id: str,
    observation: str,
    note: str,
    level: str,
    weight: float,
    current_value: float,
    baseline_value: float,
    nadir_value: float,
    days_on_treatment: int
) -> dict:
    """
    【V8.0】创建治疗响应观察项（带归因类型）
    
    参数说明：
    - report_uuid: 真实的化验单UUID（来自 lab_reports 表）
    - observation_uuid: 自动生成的观察项UUID
    """
    return {
        'observation_uuid': str(uuid_module.uuid4()),  # ✅ 新生成
        'report_uuid': report_uuid,  # ✅ 使用传入的真实UUID
        'indicator': marker_name,
        'pattern_id': pattern_id,
        'pattern_type': 'treatment_response',
        'observation': observation,
        'note': note,
        'attention_level': level,
        'score_weight': weight,
        'shap_values': {marker_name: 1.0},
        'shap_type': 'proxy',
        'context': {
            'current_value': current_value,
            'baseline_value': baseline_value,
            'nadir_value': nadir_value,
            'days_on_treatment': days_on_treatment
        }
    }


def _observe_tumor_marker_patterns(latest_data: pd.Series, 
                                   historical_data: pd.DataFrame, 
                                   ref_ranges: dict,
                                   context: dict) -> List[dict]:
    """
    观察肿瘤标志物的数据模式（已集成用户反馈）
    """
    observations = []
    all_labels = context.get('all_labels', {})
    latest_report_uuid = context.get('current_report_uuid')
    
    # 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成肿瘤标志物观察项")
        return []

    # 动态获取所有肿瘤标志物配置
    tumor_marker_configs = config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])

    for marker_config in tumor_marker_configs:
        marker_name = marker_config['name']
        reference_upper = marker_config['upper']

        if pd.isna(reference_upper) or reference_upper is None:
            continue

        # === 观察1: 连续超过参考上限 ===
        if marker_name in historical_data.columns:
            marker_history = historical_data[marker_name].dropna()

            if len(marker_history) >= 2:
                # 检查最近2次是否都超标
                if (marker_history.tail(2) > reference_upper).all():

                    # 检查反馈
                    report_uuids = historical_data.iloc[-2:]['report_uuid'].tolist()
                    labels = [all_labels.get(uuid) for uuid in report_uuids]

                    score_weight = 30
                    note = '该模式在医学文献中可能有特定含义，请向医生展示完整数据'
                    attention_level = 'high'

                    if 'benign' in labels or 'lab_error' in labels:
                        score_weight = 5
                        note = "该模式曾被标记为良性/错误，关注度已降低"
                        attention_level = 'low'
                    elif 'significant' in labels:
                        score_weight = 40
                        note = "该模式曾被标记为重要变化，关注度已提高！"
                        attention_level = 'high'

                    observations.append({
                        'observation_uuid': str(uuid_module.uuid4()),
                        'report_uuid': latest_report_uuid,
                        'indicator': marker_name,
                        'pattern_id': f'TM_CONSECUTIVE_HIGH_{marker_name.replace(" ", "_")}',
                        'pattern_type': 'consecutive_elevation',
                        'observation': f'{marker_name} 连续2次检测值超过参考上限 ({reference_upper})',
                        'note': note,
                        'attention_level': attention_level,
                        'score_weight': score_weight,
                        'shap_values': {marker_name: 1.0},
                        'shap_type': 'proxy', 
                        'context': {
                            'recent_values': marker_history.tail(2).tolist(),
                            'reference_upper': reference_upper
                        }
                    })

        # === 观察2+3 合并: 智能趋势与超标检测 ===
        if marker_name in historical_data.columns:
            marker_history = historical_data[marker_name].dropna()
            
            if len(marker_history) >= 2:
                current_value = marker_history.iloc[-1]
                prev_value = marker_history.iloc[-2]
                
                prev_date = marker_history.index[-2]
                current_date = marker_history.index[-1]
                time_diff_days = (current_date - prev_date).days
                
                # 计算变化率
                if time_diff_days > 0:
                    marker_rate = (current_value - prev_value) / time_diff_days
                else:
                    marker_rate = 0
                
                # 判断是否超标
                is_elevated = current_value > reference_upper
                
                # 判断是否有显著上升趋势
                has_rising_trend = marker_rate > 0.05
                
                # 根据不同情况生成观察
                if has_rising_trend or is_elevated:
                    description_parts = []
                    
                    # 1. 趋势部分
                    if has_rising_trend:
                        description_parts.append(f"变化率为每日 {marker_rate:.3f}，呈上升趋势")
                    
                    # 2. 超标部分
                    if is_elevated:
                        exceed_ratio = ((current_value - reference_upper) / reference_upper) * 100
                        if has_rising_trend:
                            description_parts.append(f"且当前值 ({current_value:.2f}) 超过参考上限 ({reference_upper}) {exceed_ratio:.1f}%")
                        else:
                            description_parts.append(f"当前值 ({current_value:.2f}) 超过参考上限 ({reference_upper}) {exceed_ratio:.1f}%")
                    
                    # 3. 确定风险等级
                    if has_rising_trend and is_elevated:
                        attention_level = 'high'
                        score_weight = 45
                        pattern_type = 'elevation_and_rising'
                        note = '观察提示：该指标同时呈现超标和上升趋势，请记录变化并咨询医生'
                    elif has_rising_trend:
                        attention_level = 'medium'
                        score_weight = 25
                        pattern_type = 'rate_change'
                        note = '这是基于最近两次检测计算的变化速度'
                    else:
                        attention_level = 'medium'
                        score_weight = 20
                        pattern_type = 'elevation'
                        note = '该指标的临床意义需结合其他检查综合判断'
                    
                    observation_text = f"{marker_name} " + "，".join(description_parts)
                    
                    observations.append({
                        'observation_uuid': str(uuid_module.uuid4()),
                        'report_uuid': latest_report_uuid,
                        'indicator': marker_name,
                        'pattern_id': f'TM_SMART_{marker_name.replace(" ", "_")}',
                        'pattern_type': pattern_type,
                        'observation': observation_text,
                        'note': note,
                        'attention_level': attention_level,
                        'score_weight': score_weight,
                        'shap_values': {marker_name: 1.0},
                        'shap_type': 'proxy', 
                        'context': {
                            'current_value': current_value,
                            'prev_value': prev_value,
                            'change_rate': marker_rate,
                            'reference_upper': reference_upper
                        }
                    })

    return observations



def _observe_treatment_response_patterns(
    latest_data: pd.Series,  # 保留以维持接口一致性，但此函数不需要使用
    historical_data: pd.DataFrame,
    context: dict
) -> List[dict]:
    """
    仅在阶段转换时提供温和提示（带归因类型）
    
    核心改进：
    1. 移除了原有的"RECIST 标准"推断逻辑（需要影像学数据，不适用于血液检测）
    2. 仅在"治疗→稳定"转换时提供一个通用提示
    3. 具体的治疗效果评估由 _observe_phase_transition_events 函数负责（使用 _transition_shock 特征）
    
    参数说明：
    - latest_data: (未使用) 保留以维持与其他观察函数的接口一致性
    - historical_data: 用于判断阶段转换
    - context: 包含 report_uuid 等上下文信息
    
    返回：
    - 观察项列表（仅在阶段转换时返回1个提示项）
    """

    logger.debug("RiskEngine: 正在观察'治疗反应'模式 ...")
    observations = []
    latest_report_uuid = context.get('current_report_uuid')
    
    # ✅ 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成治疗响应观察项")
        return []
    
    # 检查是否刚进入稳定期
    if len(historical_data) < 2:
        logger.debug("  > 数据点不足（<2），跳过治疗反应分析。")
        return []
    
    last_phase = historical_data['phase'].iloc[-2]
    current_phase = historical_data['phase'].iloc[-1]
    
    if not (last_phase == '强效治疗期' and current_phase == '稳定监控期'):
        logger.debug(
            f"  > 非治疗阶段转换点（{last_phase} → {current_phase}），跳过。"
        )
        return []
    
    logger.info("检测到治疗周期结束（强效治疗期 → 稳定监控期）")
    
    observations.append({
        'observation_uuid': str(uuid_module.uuid4()),
        'report_uuid': latest_report_uuid,
        'indicator': '治疗阶段',
        'pattern_id': 'TREATMENT_PHASE_END',
        'pattern_type': 'phase_transition',
        'observation': '治疗周期已结束，进入稳定监控期',
        'note': 'ℹ️ 系统将持续跟踪各项指标，如有异常会及时提示',
        'attention_level': 'info',
        'score_weight': 0,
        'shap_values': {'治疗阶段': 1.0},
        'shap_type': 'proxy',
        'context': {
            'phase_transition': f'{last_phase} → {current_phase}'
        }
    })
    
    return observations


def _observe_inflammation_patterns(latest_data: pd.Series, context: dict) -> List[dict]:
    """
    观察炎症/免疫相关综合指标的数据模式（带归因类型）
    """
    observations = []
    all_labels = context.get('all_labels', {})
    latest_report_uuid = context.get('current_report_uuid')
    current_label = all_labels.get(latest_report_uuid)

    # 如果当前报告被标记为良性或错误，大幅降低炎症指标的权重
    if current_label in ['benign', 'lab_error']:
        logger.debug("ℹ️ 炎症指标观察降权 (用户标记为良性/错误)")
        base_weight_multiplier = 0.1 # 权重降为10%
        note_suffix = " (该波动与良性事件/错误数据相关，关注度已降低)"
    elif current_label == 'significant':
        base_weight_multiplier = 1.5 # 权重提升50%
        note_suffix = " (该波动与您标记的重要变化相关，关注度已提高！)"
    else:
        base_weight_multiplier = 1.0 # 正常权重
        note_suffix = ""

    # 动态获取所有综合指标配置
    composite_configs = config.COMPOSITE_INDICATORS_CONFIG

    for indicator_key, indicator_config in composite_configs.items():
        # === 观察1: 单个指标超过阈值 ===
        if indicator_key in latest_data.index:
            current_value = latest_data[indicator_key]

            # 检查高于阈值
            if 'threshold_high' in indicator_config and pd.notna(current_value):
                threshold = indicator_config['threshold_high']

                if current_value > threshold:
                    rate_col = f'{indicator_key}_rate'
                    indicator_rate = latest_data.get(rate_col, 0)
                    trend_desc = "且呈上升趋势" if indicator_rate > 0 else "但趋势稳定"

                    observations.append({
                        'observation_uuid': str(uuid_module.uuid4()),
                        'report_uuid': latest_report_uuid,
                        'indicator': indicator_config['full_name'],
                        'pattern_id': f'COMPOSITE_HIGH_{indicator_key}',
                        'pattern_type': 'elevation',
                        'observation': f'{indicator_config["full_name"]} ({indicator_key}) 当前值 ({current_value:.2f}) 高于 {indicator_config["reference"]} {trend_desc}',
                        'note': indicator_config['clinical_note'] + note_suffix,
                        'attention_level': 'medium',
                        'score_weight': int(20 * base_weight_multiplier),
                        'shap_values': {indicator_config['full_name']: 1.0},
                        'shap_type': 'proxy',
                        'context': {
                            'current_value': current_value,
                            'reference': indicator_config['reference']
                        }
                    })

            # 检查低于阈值（如 LMR）
            if 'threshold_low' in indicator_config and pd.notna(current_value):
                threshold = indicator_config['threshold_low']

                if current_value < threshold:
                    rate_col = f'{indicator_key}_rate'
                    indicator_rate = latest_data.get(rate_col, 0)
                    trend_desc = "且呈下降趋势" if indicator_rate < 0 else "但趋势稳定"

                    observations.append({
                        'observation_uuid': str(uuid_module.uuid4()), 
                        'report_uuid': latest_report_uuid,
                        'indicator': indicator_config['full_name'],
                        'pattern_id': f'COMPOSITE_LOW_{indicator_key}',
                        'pattern_type': 'decline',
                        'observation': f'{indicator_config["full_name"]} ({indicator_key}) 当前值 ({current_value:.2f}) 低于 {indicator_config["reference"]} {trend_desc}',
                        'note': indicator_config['clinical_note'] + note_suffix,
                        'attention_level': 'medium',
                        'score_weight': int(20 * base_weight_multiplier),
                        'shap_values': {indicator_config['full_name']: 1.0},
                        'shap_type': 'proxy',
                        'context': {
                            'current_value': current_value,
                            'reference': indicator_config['reference']
                        }
                    })

    # ... (可以添加 _observe_inflammation_patterns 中其他的组合规则, 并应用 base_weight_multiplier) ...

    return observations

def _observe_nutrition_patterns(latest_data: pd.Series, 
                                historical_data: pd.DataFrame,
                                context: dict) -> List[dict]:
    """
    观察营养/免疫相关指标 （带归因类型）
    """
    observations = []
    all_labels = context.get('all_labels', {})
    latest_report_uuid = context.get('current_report_uuid')
    
    # 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成营养指标观察项")
        return []
    
    current_label = all_labels.get(latest_report_uuid)

    # 营养指标通常是慢变量，受“良性”事件（如感冒）影响小，但受“重要”事件影响大
    if current_label == 'significant':
        base_weight_multiplier = 1.5
        note_suffix = " (该波动与您标记的重要变化相关，关注度已提高)"
    else:
        base_weight_multiplier = 1.0
        note_suffix = ""

    nutrition_indicators = config.NUTRITION_INDICATORS_CONFIG

    for indicator_name, indicator_config in nutrition_indicators.items():
        if indicator_name not in historical_data.columns:
            continue

        threshold_low = indicator_config['threshold_low']

        # === 观察: 持续低于阈值 ===
        indicator_history = historical_data[indicator_name].dropna()
        if len(indicator_history) >= 3:
            if (indicator_history.tail(3) < threshold_low).all():
                observations.append({
                    'observation_uuid': str(uuid_module.uuid4()),
                    'report_uuid': latest_report_uuid,
                    'indicator': indicator_name,
                    'pattern_id': f'NUTRITION_PERSISTENT_LOW_{indicator_name.replace(" ", "_")}',
                    'pattern_type': 'persistent_low',
                    'observation': f'{indicator_name} 连续3次低于 {threshold_low}',
                    'note': '持续低水平需要关注营养摄入和吸收情况' + note_suffix,
                    'attention_level': 'medium',
                    'score_weight': int(25 * base_weight_multiplier),
                    'shap_values': {indicator_name: 1.0},
                    'shap_type': 'proxy',
                    'context': {
                        'recent_values': indicator_history.tail(3).tolist(),
                        'threshold': threshold_low
                    }
                })

    return observations




def _calculate_indicator_volatility(
    historical_data: pd.DataFrame,
    indicator_name: str,
    min_samples: int = 5
) -> float:
    """
    计算指标的历史波动性（变异系数 CV）
    
    :return: CV 值（0-1之间，越大越波动）
    """
    if indicator_name not in historical_data.columns:
        return 0.3  # 默认中等波动
    
    values = historical_data[indicator_name].dropna()
    
    if len(values) < min_samples:
        return 0.3
    
    mean_val = values.mean()
    std_val = values.std()
    
    if abs(mean_val) < config.EPSILON:
        return 0.3
    
    cv = std_val / abs(mean_val)
    return np.clip(cv, 0.1, 1.0)  # 限制在 [0.1, 1.0] 范围


def _classify_trend_type(diffs: pd.Series) -> str:
    """
    【辅助函数】分类趋势类型
    
    :param diffs: 差分序列（pd.Series.diff()）
    :return: 趋势类型字符串
    """
    if diffs.empty or len(diffs) < 2:
        return "数据不足"
    
    mean_diff = diffs.mean()
    std_diff = diffs.std()
    
    if abs(mean_diff) < config.EPSILON:
        return "围绕基线波动"
    
    cv = std_diff / abs(mean_diff)
    
    if cv < 0.3:
        return "持续趋势"
    elif cv > 0.8:
        return "围绕基线波动"
    else:
        return "混合模式"


def _get_dynamic_dtw_constraints(
    indicator_name: str,
    historical_data: pd.DataFrame,
    matched_snippet: pd.Series,
    current_snippet: pd.Series,
    context: dict = None
) -> tuple:
    """
    【V6.3 增强版】为 DTW 临床过滤器计算个性化的约束参数
    
    核心策略：
    1. 基于指标类型的基准约束（从配置读取）
    2. 基于患者历史波动性的调整
    3. 基于匹配片段质量的微调
    4. 基于治疗阶段的调整
    5. 基于趋势类型的验证
    6. 基于历史标签的动态调整
    
    :return: (range_tolerance, monotonicity_tolerance)
    """
    
    # ========================================
    # 第1层：基于指标类型的基准约束
    # ========================================
    
    # 从配置中查找指标类型
    indicator_category = None
    for report_type, indicators_list in config.LAB_REPORT_CONFIG.items():
        for ind_config in indicators_list:
            if ind_config['name'] == indicator_name:
                indicator_category = ind_config['category']
                break
        if indicator_category:
            break
    
    # 检查是否为综合指标
    if indicator_name in config.COMPOSITE_INDICATORS:
        indicator_category = 'COMPOSITE'
    
    # 基准约束字典（根据临床经验设定）
    BASE_CONSTRAINTS = {
        # 肿瘤标志物：波动较小，约束较严格
        'TUMOR_MARKER': {
            'range_tolerance': 0.8,      # 振幅差异 ≤ 80%
            'monotonicity_tolerance': 0.6  # 趋势差异 ≤ 60%
        },
        # 炎症指标：生理性波动大，约束宽松
        'INFLAMMATION': {
            'range_tolerance': 1.5,
            'monotonicity_tolerance': 0.8
        },
        # 血常规：波动中等
        'BLOOD_ROUTINE': {
            'range_tolerance': 1.0,
            'monotonicity_tolerance': 0.7
        },
        # 综合指标（NLR、PLR等）：波动大
        'COMPOSITE': {
            'range_tolerance': 1.8,
            'monotonicity_tolerance': 0.9
        },
        # 默认约束（保守策略）
        'DEFAULT': {
            'range_tolerance': 1.2,
            'monotonicity_tolerance': 0.7
        }
    }
    
    # 炎症指标细分逻辑
    if indicator_category == 'COMPOSITE':
        # 根据具体指标细化约束
        if indicator_name in ['NLR', 'SII', 'SIRI']:
            # 中性粒细胞主导的炎症指标：允许更大的波动
            base_range_tol = 3.5        # 350% 基准
            base_mono_tol = 1.0         # 不强制趋势一致
            max_range_tol = 10.0        # 上限1000%
            logger.debug(f"  DTW: {indicator_name} 采用高波动炎症指标约束")
            
        elif indicator_name in ['PLR', 'LMR']:
            # 血小板/淋巴细胞指标：中等波动
            base_range_tol = 2.5
            base_mono_tol = 0.9
            max_range_tol = 6.0
            logger.debug(f"  DTW: {indicator_name} 采用中等波动炎症指标约束")
            
        else:
            # 其他综合指标：使用原配置
            base_range_tol = 1.8
            base_mono_tol = 0.9
            max_range_tol = 5.0
            logger.debug(f"  DTW: {indicator_name} 采用默认综合指标约束")
    else:
        # 非综合指标：使用原有逻辑
        base_constraints = BASE_CONSTRAINTS.get(
            indicator_category, 
            BASE_CONSTRAINTS['DEFAULT']
        )
        base_range_tol = base_constraints['range_tolerance']
        base_mono_tol = base_constraints['monotonicity_tolerance']
        
        # 为非综合指标设置 max_range_tol（保持原逻辑）
        if indicator_category == 'TUMOR_MARKER':
            max_range_tol = 2.5
        else:
            max_range_tol = 5.0
    
    logger.debug(
        f"  DTW 约束 ({indicator_name}): "
        f"类别={indicator_category}, "
        f"基准约束=[{base_range_tol:.2f}, {base_mono_tol:.2f}]"
    )
    
    # ========================================
    # 第2层：基于患者历史波动性的调整
    # ========================================
    
    patient_cv = _calculate_indicator_volatility(
        historical_data, 
        indicator_name, 
        min_samples=5
    )
    
    cv_adjustment = 0.8 + (patient_cv * 0.7)  # [0.8, 1.5]
    
    adjusted_range_tol = base_range_tol * cv_adjustment
    adjusted_mono_tol = base_mono_tol * cv_adjustment
    
    logger.debug(
        f"  DTW 约束 ({indicator_name}): "
        f"患者CV={patient_cv:.2f}, "
        f"调整后=[{adjusted_range_tol:.2f}, {adjusted_mono_tol:.2f}]"
    )
    
    # ========================================
    # 第3层：基于匹配片段时间跨度的微调
    # ========================================
    
    matched_days = (matched_snippet.index.max() - matched_snippet.index.min()).days
    current_days = (current_snippet.index.max() - current_snippet.index.min()).days
    
    if current_days > 0:
        time_span_ratio = matched_days / current_days
        
        # 根据治疗阶段调整
        if time_span_ratio > 2.0 or time_span_ratio < 0.5:
            # 获取当前治疗阶段
            current_phase = context.get('current_phase_tag') if context else None
            
            if current_phase == '强效治疗期':
                # 治疗期：采样频率变化可能意味着病情变化，应该更严格
                span_adjustment = 0.9  # 收紧 10%
                logger.debug(
                    f"  DTW 约束 ({indicator_name}): "
                    f"时间跨度差异大 ({time_span_ratio:.1f}x) + 治疗期，"
                    f"收紧约束到 [{adjusted_range_tol * span_adjustment:.2f}, "
                    f"{adjusted_mono_tol * span_adjustment:.2f}]"
                )
            else:
                # 非治疗期：正常放宽
                span_adjustment = 1.2  # 放宽 20%
                logger.debug(
                    f"  DTW 约束 ({indicator_name}): "
                    f"时间跨度差异大 ({time_span_ratio:.1f}x)，"
                    f"放宽到 [{adjusted_range_tol * span_adjustment:.2f}, "
                    f"{adjusted_mono_tol * span_adjustment:.2f}]"
                )
            
            adjusted_range_tol *= span_adjustment
            adjusted_mono_tol *= span_adjustment
    
    # ========================================
    # 第4层：基于趋势类型的验证（新增）
    # ========================================
    
    # 计算趋势类型
    matched_diffs = matched_snippet.diff().dropna()
    current_diffs = current_snippet.diff().dropna()
    
    matched_trend = _classify_trend_type(matched_diffs)
    current_trend = _classify_trend_type(current_diffs)
    
    logger.debug(
        f"  DTW 约束 ({indicator_name}): "
        f"历史趋势={matched_trend}, 当前趋势={current_trend}"
    )
    
    # 如果趋势类型完全不同，提高约束（更严格）
    if matched_trend == "持续趋势" and current_trend == "围绕基线波动":
        trend_adjustment = 0.8  # 收紧 20%
        adjusted_range_tol *= trend_adjustment
        adjusted_mono_tol *= trend_adjustment
        
        logger.debug(
            f"  DTW 约束 ({indicator_name}): "
            f"趋势类型不匹配（历史持续 vs 当前波动），"
            f"收紧到 [{adjusted_range_tol:.2f}, {adjusted_mono_tol:.2f}]"
        )
    
    elif matched_trend == "围绕基线波动" and current_trend == "持续趋势":
        trend_adjustment = 0.8
        adjusted_range_tol *= trend_adjustment
        adjusted_mono_tol *= trend_adjustment
        
        logger.debug(
            f"  DTW 约束 ({indicator_name}): "
            f"趋势类型不匹配（历史波动 vs 当前持续），"
            f"收紧到 [{adjusted_range_tol:.2f}, {adjusted_mono_tol:.2f}]"
        )
    
    # ========================================
    # 第5层：基于历史标签的动态调整（新增）
    # ========================================
    
    if context:
        all_labels = context.get('all_labels', {})

    if context:
        all_labels = context.get('all_labels', {})
        
        # 增加容错：检查 report_uuid 列是否存在
        if 'report_uuid' not in historical_data.columns:
            logger.debug(
                f"  DTW 约束 ({indicator_name}): "
                f"historical_data 中没有 'report_uuid' 列，跳过标签调整"
            )
        else:
            # 获取匹配片段涉及的所有 UUID
            try:
                # 确保 matched_snippet 的索引在 historical_data 中
                matched_indices = matched_snippet.index
                valid_indices = matched_indices.intersection(historical_data.index)
                
                if len(valid_indices) > 0:
                    matched_uuids = historical_data.loc[valid_indices, 'report_uuid'].dropna().unique()
                    matched_labels = [all_labels.get(uuid) for uuid in matched_uuids if uuid]
                    
                    # 统计标签分布
                    significant_count = matched_labels.count('significant')
                    benign_count = matched_labels.count('benign')
                    
                    logger.debug(
                        f"  DTW 约束 ({indicator_name}): "
                        f"历史标签统计: significant={significant_count}, benign={benign_count}"
                    )
                    
                    # 如果历史被多次标记为重要，收紧约束
                    if significant_count > benign_count and significant_count > 0:
                        label_adjustment = 0.8  # 收紧 20%
                        adjusted_range_tol *= label_adjustment
                        adjusted_mono_tol *= label_adjustment
                        
                        logger.debug(
                            f"  DTW 约束 ({indicator_name}): "
                            f"历史多次标记为重要变化，收紧到 "
                            f"[{adjusted_range_tol:.2f}, {adjusted_mono_tol:.2f}]"
                        )
                    
                    # 如果历史多次标记为良性，放宽约束
                    elif benign_count > significant_count and benign_count > 0:
                        label_adjustment = 1.2  # 放宽 20%
                        adjusted_range_tol *= label_adjustment
                        adjusted_mono_tol *= label_adjustment
                        
                        logger.debug(
                            f"  DTW 约束 ({indicator_name}): "
                            f"历史多次标记为良性，放宽到 "
                            f"[{adjusted_range_tol:.2f}, {adjusted_mono_tol:.2f}]"
                        )
        
            except Exception as label_e:
                logger.warning(
                    f"  DTW 约束 ({indicator_name}): "
                    f"提取历史标签失败: {label_e}"
                )
    
    # ========================================
    # 安全护栏：确保约束在合理范围内
    # ========================================
    
    max_mono_tol = 1.0  # 单调性上限保持不变
    
    # 应用上限
    final_range_tol = min(adjusted_range_tol, max_range_tol)
    final_mono_tol = min(adjusted_mono_tol, max_mono_tol)
    
    logger.debug(
        f"  DTW 约束 ({indicator_name}): "
        f"最终约束=[{final_range_tol:.2f}, {final_mono_tol:.2f}] "
        f"(上限: range={max_range_tol:.1f}, mono={max_mono_tol:.1f})"
    )
    
    return final_range_tol, final_mono_tol


def _clinical_dtw_filter(
    matched_snippet: pd.Series, 
    current_snippet: pd.Series,
    indicator_name: str = None, 
    historical_data: pd.DataFrame = None,
    context: dict = None
) -> tuple:
    """
    【V6.2 动态约束版】对DTW的匹配结果施加"生物学约束"
    
    核心改进：
    1. 移除硬编码的约束参数
    2. 根据指标类型、患者特征、数据质量动态计算约束
    3. 保留原有的约束逻辑（振幅 + 单调性）
    
    :param matched_snippet: 历史匹配片段
    :param current_snippet: 当前查询片段
    :param indicator_name: 指标名称（用于动态约束计算）
    :param historical_data: 完整历史数据（用于波动性分析）
    :return: (是否通过, 失败原因)
    """
    logger.debug("  DTW: 正在应用临床约束过滤器 (V6.2 动态版)...")
    
    # ========================================
    # 动态计算约束参数
    # ========================================
    if indicator_name and historical_data is not None:
        range_tolerance, monotonicity_tolerance = _get_dynamic_dtw_constraints(
            indicator_name,
            historical_data,
            matched_snippet,
            current_snippet,
            context
        )
    else:
        # 兜底：使用保守的默认值
        range_tolerance = 1.2
        monotonicity_tolerance = 0.7
        logger.warning("DTW: 无法获取指标信息，使用默认约束。")
    
    try:
        # ========================================
        # 约束 1：绝对值范围 (振幅)
        # ========================================
        matched_range = matched_snippet.max() - matched_snippet.min()
        current_range = current_snippet.max() - current_snippet.min()
        
        if abs(current_range) < config.EPSILON:
            logger.debug("  DTW 约束失败: 当前范围为0。")
            return False, "数值范围为0"

        range_diff_ratio = abs(matched_range - current_range) / current_range
        if range_diff_ratio > range_tolerance:
            logger.debug(
                f"  DTW 约束失败: 振幅范围差异过大 "
                f"({range_diff_ratio:.1%} > {range_tolerance:.1%})。"
            )
            return False, "数值范围差异过大"
        
        # ========================================
        # 约束 2：单调性 (趋势方向)
        # ========================================
        matched_diffs = matched_snippet.diff().dropna()
        current_diffs = current_snippet.diff().dropna()
        
        if matched_diffs.empty or current_diffs.empty:
            logger.debug("  DTW 约束失败: 无法计算单调性 (数据点不足)。")
            return False, "趋势数据不足"

        # 计算上升趋势的占比
        matched_monotonic = (matched_diffs > 0).sum() / len(matched_diffs)
        current_monotonic = (current_diffs > 0).sum() / len(current_diffs)
        
        mono_diff = abs(matched_monotonic - current_monotonic)
        if mono_diff > monotonicity_tolerance:
            logger.debug(
                f"  DTW 约束失败: 趋势方向不一致 "
                f"(差异: {mono_diff:.1%} > {monotonicity_tolerance:.1%})。"
            )
            return False, "趋势方向不一致"

        logger.debug("  DTW: ✅ 通过临床约束。")
        return True, "通过临床约束"
        
    except Exception as e:
        logger.warning(f"  DTW 临床约束过滤器执行失败: {e}", exc_info=True)
        return False, "过滤器异常"
    

def _observe_historical_similarity(latest_data: pd.Series,
                                   historical_data: pd.DataFrame,
                                   context: dict) -> List[dict]:
    """
    观察历史相似模式（DTW + 主动学习请求 + 归因类型）
    
    核心改进：
    1. 移除硬编码的时间容忍度，使用 DTW 内部的自适应策略
    2. 当匹配到未标记的历史模式时，主动请求用户标记
    3. 相似度阈值 30%（更灵活的匹配）
    """
    observations = []
    all_labels = context.get('all_labels', {})
    latest_report_uuid = context.get('current_report_uuid')
    patient_id = context.get('patient_id')
    
    # ✅ 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成历史相似性观察项")
        return []

    key_indicators = []
    tumor_marker_configs = config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
    key_indicators.extend([marker['name'] for marker in tumor_marker_configs])
    key_indicators.extend(config.COMPOSITE_INDICATORS)

    for indicator in key_indicators:
        if indicator not in historical_data.columns:
            continue

        full_history = historical_data[indicator].dropna()

        if len(full_history) < 10:
            continue

        snippet_len = min(3, len(full_history) // 3)
        if snippet_len < 3: 
            continue

        current_snippet = full_history.tail(snippet_len)

        try:
            best_match, min_dist = analysis_engine.find_most_similar_pattern_dtw(
                current_snippet, 
                full_history, 
                snippet_len
            )

            if best_match is not None and min_dist != float('inf'):

                # 临床约束
                pass_filter, reason = _clinical_dtw_filter(
                    best_match, 
                    current_snippet,
                    indicator_name=indicator,
                    historical_data=historical_data,
                    context=context
                )

                if not pass_filter:
                    logger.info(
                        f"RiskEngine (DTW): {indicator} 匹配到 {best_match.index.min().strftime('%Y-%m')}. "
                        f"但未通过临床约束: {reason}。已跳过。"
                    )
                    continue
         
                similarity = 1 / (1 + min_dist + 1e-6)

                if similarity > config.DTW_SIMILARITY_THRESHOLD:
                    logger.debug(
                        f"DTW: {indicator} 相似度={similarity:.0%} "
                        f"(阈值={config.DTW_SIMILARITY_THRESHOLD:.0%})"
                    )

                    # 获取历史标签
                    match_uuids = historical_data.loc[best_match.index, 'report_uuid'].unique()
                    match_labels = [all_labels.get(uuid) for uuid in match_uuids if uuid]
                    
                    is_historical_unlabeled = not any(
                        label in ['significant', 'benign', 'lab_error'] 
                        for label in match_labels
                    )

                    # 构建基础观察项
                    obs_dict = {
                        'observation_uuid': str(uuid_module.uuid4()),  # ✅ 新生成
                        'report_uuid': latest_report_uuid,  # ✅ 真实UUID
                        'indicator': indicator,
                        'pattern_id': f'DTW_SIMILARITY_{indicator.replace(" ", "_")}',
                        'pattern_type': 'historical_pattern_match',
                        'observation': f'{indicator} 最近变化模式与 {best_match.index.min().strftime("%Y年%m月")} 时期相似',
                        'shap_values': {indicator: 1.0},
                        'shap_type': 'proxy',  # ✅ 新增
                        'context': {
                            'similarity_score': similarity,
                            'matched_date': best_match.index.min().strftime("%Y年%m月")
                        }
                    }

                    # 主动学习请求
                    if is_historical_unlabeled and similarity > config.DTW_HIGH_SIMILARITY_THRESHOLD:
                        logger.info(
                            f"DTW: {indicator} 相似度 {similarity:.0%}，"
                            f"触发历史标签请求（历史模式未标记）"
                        )
                        
                        try:
                            earliest_idx = best_match.index.min()
                            
                            if earliest_idx not in historical_data.index:
                                logger.warning(
                                    f"DTW: 无法找到历史日期 {earliest_idx} 对应的记录，"
                                    f"跳过标签请求"
                                )
                                historical_uuid_to_label = None
                            else:
                                historical_uuid_to_label = historical_data.loc[
                                    earliest_idx, 'report_uuid'
                                ]
                            
                            if historical_uuid_to_label and pd.notna(historical_uuid_to_label):
                                obs_dict['unified_feedback_request'] = {
                                    'uuid': historical_uuid_to_label,
                                    'date_str': earliest_idx.strftime("%Y年%m月%d日"),
                                    'indicator': indicator,
                                    'similarity_pct': int(similarity * 100),
                                    'matched_date_str': best_match.index.min().strftime("%Y年%m月"),
                                    'type': 'similarity_with_label',
                                    'matched_date_obj': best_match.index.min()
                                }
                                
                                obs_dict['note'] = (
                                    f'与 {obs_dict["context"]["matched_date"]} 相似。'
                                    f'帮助 AI 理解：那次是良性波动还是重要变化？'
                                )
                                obs_dict['attention_level'] = 'low'
                                obs_dict['score_weight'] = 10
                            else:
                                obs_dict['note'] = f'您可以回顾 {obs_dict["context"]["matched_date"]} 时期的记录'
                                obs_dict['attention_level'] = 'info'
                                obs_dict['score_weight'] = 5
                                
                        except Exception as extract_e:
                            logger.error(
                                f"DTW: 提取历史 UUID 失败: {extract_e}",
                                exc_info=True
                            )
                            obs_dict['note'] = f'您可以回顾 {obs_dict["context"]["matched_date"]} 时期的记录'
                            obs_dict['attention_level'] = 'info'
                            obs_dict['score_weight'] = 5
                    
                    else:
                        # 已标记或相似度不够高
                        if 'significant' in match_labels:
                            obs_dict['score_weight'] = 40
                            obs_dict['note'] = f'与 {best_match.index.min().strftime("%Y年%m月")} 相似，您曾标记为"重要变化"！'
                            obs_dict['attention_level'] = 'high'
                        
                        elif 'benign' in match_labels:
                            # 提供安抚信息，但保持低权重
                            obs_dict['score_weight'] = 5
                            obs_dict['note'] = f'✅ 当前模式与 {best_match.index.min().strftime("%Y年%m月")} 相似，您曾标记为"良性波动"'
                            obs_dict['attention_level'] = 'info'
                            logger.debug(
                                f"DTW: {indicator} 匹配到良性历史模式，"
                                f"生成安抚性提示（权重={obs_dict['score_weight']}）"
                            )
                        
                        elif 'lab_error' in match_labels:
                            # 错误数据也生成提示，但明确标注
                            obs_dict['score_weight'] = 0  # 不影响总分
                            obs_dict['note'] = f'ℹ️ 当前模式与 {best_match.index.min().strftime("%Y年%m月")} 相似，但那次被标记为"数据错误"'
                            obs_dict['attention_level'] = 'info'
                            logger.info(
                                f"DTW: {indicator} 匹配到错误数据历史，"
                                f"生成提示但权重为0"
                            )
                        
                        else:
                            # 匹配到未标记的历史（相似度不够高）
                            obs_dict['score_weight'] = 5
                            obs_dict['note'] = f'您可以回顾 {best_match.index.min().strftime("%Y年%m月")} 时期的记录'
                            obs_dict['attention_level'] = 'info'

                    observations.append(obs_dict)

        except Exception as e:
            logger.warning(f"ℹ️ {indicator} 模式匹配跳过: {e}", exc_info=True)

    return observations


def _observe_progression_pattern(historical_data: pd.DataFrame, context: dict) -> List[dict]:
    """
    通过分析"标签"的变化序列，在后端静默推断病情进展（带归因类型）
    """
    observations = []
    all_labels = context.get('all_labels', {})
    latest_report_uuid = context.get('current_report_uuid')
    
    # 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成病程模式观察项")
        return []

    if 'phase' not in historical_data.columns:
        logger.warning("病程模式分析跳过：找不到 'phase' 列。")
        return []

    # 使用 groupby 和 shift 来查找真正的“状态转变”
    # 1. 按时间排序
    sorted_phases = historical_data.sort_index()['phase']

    # 2. 压缩连续相同的 phase
    phase_sequence = sorted_phases.groupby(
        (sorted_phases != sorted_phases.shift()).cumsum()
    ).first().tolist()

    logger.debug(f"病程模式分析：找到的压缩序列: {' -> '.join(phase_sequence)}")

    try:
        # 找到第一个 "稳定监控期"
        first_stable_index = phase_sequence.index("稳定监控期")
        # 只看这之后的序列
        sub_sequence = phase_sequence[first_stable_index:]

        # 如果 "稳定" 之后又出现了 "治疗"
        if "强效治疗期" in sub_sequence:

            # 检查 "稳定 -> 治疗" 这个模式在子序列中出现了多少次
            transition_count = 0
            for i in range(len(sub_sequence) - 1):
                if sub_sequence[i] == "稳定监控期" and sub_sequence[i+1] == "强效治疗期":
                    transition_count += 1

            if transition_count == 0:
                logger.debug("病程模式：未找到 'S -> T' 转换，跳过。")
                return []

            # 找到最后一个 "强效治疗期" 的数据行
            last_treatment_rows = historical_data[historical_data['phase'] == '强效治疗期']
            if last_treatment_rows.empty:
                return [] 

            last_treatment_uuid = last_treatment_rows.iloc[-1]['report_uuid']
            last_treatment_label = all_labels.get(last_treatment_uuid)

            score_weight = 50 + (transition_count * 10) # 每多一次S->T转换，权重增加
            note = f'系统已观察到此模式 ({transition_count} 次"稳定->治疗"转换)，请将您的完整情况告知医生。'
            level = 'high'

            if last_treatment_label == 'benign':
                score_weight = 10 
                note = '观察到 稳定期->治疗期 模式，但最近治疗期曾被标记为良性相关。'
                level = 'medium'
            elif last_treatment_label == 'lab_error':
                score_weight = 0
                note = '观察到 稳定期->治疗期 模式，但被标记为数据错误。'
                level = 'low'

            logger.info(
                f"RiskEngine (Progression): 触发 S->T 模式! "
                f"Transitions={transition_count}. "
                f"Label='{last_treatment_label}'. "
                f"Final Weight={score_weight}"
            )

            if score_weight > 0:
                observations.append({
                    'observation_uuid': str(uuid_module.uuid4()),
                    'report_uuid': latest_report_uuid,
                    'indicator': '病程模式',
                    'pattern_id': 'INFERRED_PROGRESSION_PATTERN',
                    'pattern_type': 'Inferred Progression Pattern',
                    'observation': f'观察到在稳定监控期后，再次进入强效治疗期的数据模式 (共 {transition_count} 次)',
                    'note': note,
                    'attention_level': level,
                    'score_weight': score_weight,
                    'shap_values': {'病程模式': 1.0},
                    'shap_type': 'proxy',
                    'context': {
                        'phase_sequence': ' -> '.join(phase_sequence),
                        'transition_count': transition_count
                    }
                })
    except ValueError:
        logger.debug("病程模式：未找到'稳定监控期'，跳过推断。")
    except Exception as e:
        logger.error(f"❌ 病程模式分析失败: {e}", exc_info=True)

    return observations


def _observe_phase_transition_events(
    latest_data: pd.Series,
    historical_data: pd.DataFrame,
    context: dict
) -> List[dict]:
    """
    观察阶段转换事件（基于 _transition_shock 特征 + 归因类型）
    
    目的：识别"治疗→稳定"转换时的指标剧烈变化（通常是好事）
    """
    observations = []
    latest_report_uuid = context.get('current_report_uuid')
    
    # ✅ 验证UUID
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成阶段转换观察项")
        return []
    
    # 检查当前点是否是阶段边界
    if len(historical_data) < 2:
        return []
    
    last_phase = historical_data['phase'].iloc[-2]
    current_phase = historical_data['phase'].iloc[-1]
    
    # 只关注"治疗→稳定"转换
    if not (last_phase == '强效治疗期' and current_phase == '稳定监控期'):
        return []
    
    logger.info("检测到阶段转换：治疗期 → 稳定期")
    
    # 2. 获取所有肿瘤标志物的 transition_shock
    tumor_markers = [
        item['name'] for item in config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
    ]
    
    for marker in tumor_markers:
        shock_col = f'{marker}_transition_shock'
        
        if shock_col not in latest_data.index or pd.isna(latest_data[shock_col]):
            continue
        
        shock_value = latest_data[shock_col]
        
        # 3. 判断：显著下降 = 治疗有效
        if shock_value < -5.0:  # 每天下降超过 5 单位
            observations.append({
                'observation_uuid': str(uuid_module.uuid4()),
                'report_uuid': latest_report_uuid,
                'indicator': marker,
                'pattern_id': f'TRANSITION_RESPONSE_{marker}',
                'pattern_type': 'treatment_response',
                'observation': f'{marker} 在治疗结束时显著下降（{shock_value:.1f}/天）',
                'note': '观察提示：这是治疗反应较好的数据现象，后续安排请咨询医生',
                'attention_level': 'low',
                'score_weight': 5,
                'shap_values': {marker: 1.0},
                'shap_type': 'proxy',
                'context': {
                    'shock_value': shock_value,
                    'phase_transition': f'{last_phase} → {current_phase}'
                }
            })
        
        # 4. 判断：上升/不变 = 疗效欠佳
        elif shock_value > -1.0:
            observations.append({
                'observation_uuid': str(uuid_module.uuid4()),
                'report_uuid': latest_report_uuid,
                'indicator': marker,
                'pattern_id': f'TRANSITION_POOR_RESPONSE_{marker}',
                'pattern_type': 'poor_response',
                'observation': f'{marker} 治疗后未显著下降（{shock_value:.1f}/天）',
                'note': '观察提示：请咨询医生评估疗效',
                'attention_level': 'medium',
                'score_weight': 40,
                'shap_values': {marker: 1.0},
                'shap_type': 'proxy',
                'context': {
                    'shock_value': shock_value,
                    'phase_transition': f'{last_phase} → {current_phase}'
                }
            })
    
    return observations


def _observe_sparse_longitudinal_predictions(
    historical_data: pd.DataFrame,
    context: dict
) -> List[dict]:
    """
    用稀疏纵向预测引擎观察小样本轨迹。

    该模块只预测“未来化验值/可靠变化概率”，不把结果解释为病情诊断。
    """

    observations = []
    latest_report_uuid = context.get('current_report_uuid')
    if not latest_report_uuid:
        logger.error("❌ 缺少 current_report_uuid，无法生成小样本纵向预测观察项")
        return []

    try:
        sparse_report = predictive_ensemble_engine.analyze_predictive_ensemble(
            historical_data,
            horizon_days=(30, 60, 90),
            max_indicators=8,
        )
    except Exception as exc:
        logger.error(f"稀疏纵向预测失败: {exc}", exc_info=True)
        return []

    for item in sparse_report.get("top_observations", [])[:3]:
        classification = item.get("classification", "不足以判断")
        if classification not in {"需要复核的数据变化", "可观察预测"}:
            continue

        attention_level = item.get("attention_level", "info")
        if attention_level == "medium":
            score_weight = 18
        elif attention_level == "low":
            score_weight = 8
        else:
            score_weight = 3

        observations.append({
            'observation_uuid': str(uuid_module.uuid4()),
            'report_uuid': latest_report_uuid,
            'indicator': item.get("indicator", "小样本纵向预测"),
            'pattern_id': f"SPARSE_LONGITUDINAL_{str(item.get('indicator', 'NA')).replace(' ', '_')}",
            'pattern_type': 'sparse_longitudinal_prediction',
            'observation': item.get("observation", ""),
            'note': item.get(
                "note",
                "这是基于稀疏纵向数据的概率提示，请咨询医生综合判断。"
            ),
            'attention_level': attention_level,
            'score_weight': score_weight,
            'shap_values': {item.get("indicator", "小样本纵向预测"): 1.0},
            'shap_type': 'proxy',
            'context': {
                'classification': classification,
                'priority_score': item.get("priority_score", 0.0),
                'method': sparse_report.get("method"),
            }
        })

    return observations


# ========================================
# 3. 汇总与展示（粘贴回来的）
# ========================================

def _calculate_attention_score(all_observations: List[dict]) -> tuple:
    """
    计算"关注度评分"
    """
    total_score = sum(obs['score_weight'] for obs in all_observations)

    if total_score >= 70:
        level = 'high_attention'
        level_desc = '多项数据显著波动'
        color = 'critical'
    elif total_score >= 40:
        level = 'moderate_attention'
        level_desc = '部分数据出现波动'
        color = 'high'
    elif total_score >= 20:
        level = 'mild_attention'
        level_desc = '少量数据波动'
        color = 'medium'
    else:
        level = 'stable'
        level_desc = '数据整体稳定'
        color = 'low'

    return level, level_desc, total_score, color

def _generate_summary_statement(attention_level: str, all_observations: List[dict]) -> str:
    """
    生成纯事实性的摘要
    """
    if not all_observations:
        return "当前数据相对稳定，未观察到明显的统计学异常模式。"

    high_attention = [o for o in all_observations if o.get('attention_level') == 'high']
    medium_attention = [o for o in all_observations if o.get('attention_level') == 'medium']

    if high_attention:
        indicators = sorted(list(set(o['indicator'] for o in high_attention)))
        return f"观察到 {len(indicators)} 个指标出现显著变化：{', '.join(indicators[:3])}{'等' if len(indicators) > 3 else ''}。"

    if medium_attention:
        indicators = sorted(list(set(o['indicator'] for o in medium_attention)))
        return f"观察到 {len(indicators)} 个指标出现轻度波动：{', '.join(indicators[:2])}{'等' if len(indicators) > 2 else ''}。"

    return "整体数据趋势保持平稳。"





# ========================================
# 4. 主入口函数（已修改）
# ========================================

def observe_health_data_patterns(latest_data: pd.Series, 
                                historical_data: pd.DataFrame, 
                                ref_ranges: dict,
                                context: dict) -> dict: 
    """
    健康数据模式观察主入口

    ⚠️ 重要: 本函数仅提供数据统计观察，不构成任何医疗建议
    """

    logger.info("\n" + "="*60)
    logger.info("📊 开始健康数据模式分析 (V2 - 集成反馈)...")
    logger.info("="*60)

    all_observations = []
    
    # --- V-Clinician 修正：推断"当前治疗周期"的起始日期 ---
    # 这是实现新的"治疗反应"逻辑所必需的
    current_phase = context.get('current_phase_tag')
    if current_phase == '强效治疗期':
        try:
            phases_series = historical_data['phase']
            
            # 找出在此日期之前，最后一次"非治疗期"的索引
            last_stable_idx = phases_series[
                (phases_series != '强效治疗期') & (phases_series.index < latest_data.name)
            ].last_valid_index()
            
            if last_stable_idx is None:
                # 如果从未有过稳定期，则治疗开始于本周期的第一个日期
                current_treatment_start_date = phases_series[phases_series == '强效治疗期'].index.min()
            else:
                # 否则，治疗开始于"最后稳定日"之后的第一个记录
                current_treatment_start_date = phases_series[phases_series.index > last_stable_idx].index.min()
            
            context['current_treatment_start_date'] = current_treatment_start_date
            logger.info(f"推断出当前治疗周期开始于: {current_treatment_start_date}")
            
        except Exception as e:
            logger.warning(f"推算治疗起始日期失败: {e}")
            context['current_treatment_start_date'] = None

    # 1.0 病程进展模式 (最高优先级)
    try:
        prog_obs = _observe_progression_pattern(historical_data, context)
        all_observations.extend(prog_obs)
        logger.info(f"  - 病程模式: {len(prog_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 病程模式分析失败: {e}", exc_info=True)

    # 1.2 阶段转换事件
    try:
        transition_obs = _observe_phase_transition_events(latest_data, historical_data, context)
        all_observations.extend(transition_obs)
        logger.info(f"  - 阶段转换: {len(transition_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 阶段转换分析失败: {e}", exc_info=True)

    # 1.3 肿瘤标志物模式
    try:
        tm_obs = _observe_tumor_marker_patterns(latest_data, historical_data, ref_ranges, context)
        all_observations.extend(tm_obs)
        logger.info(f"  - 肿瘤标志物: {len(tm_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 肿瘤标志物分析失败: {e}", exc_info=True)

    # 1.4 治疗反应模式
    try:
        tr_obs = _observe_treatment_response_patterns(latest_data, historical_data, context)
        all_observations.extend(tr_obs)
        logger.info(f"  - 治疗反应: {len(tr_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 治疗反应分析失败: {e}", exc_info=True)

    # 1.45 小样本纵向预测
    try:
        sparse_obs = _observe_sparse_longitudinal_predictions(historical_data, context)
        all_observations.extend(sparse_obs)
        logger.info(f"  - 小样本纵向预测: {len(sparse_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 小样本纵向预测失败: {e}", exc_info=True)

    # 1.5 炎症指标模式
    try:
        inflam_obs = _observe_inflammation_patterns(latest_data, context)
        all_observations.extend(inflam_obs)
        logger.info(f"  - 炎症指标: {len(inflam_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 炎症指标分析失败: {e}", exc_info=True)

    # 1.6 营养/免疫模式
    try:
        nutrition_obs = _observe_nutrition_patterns(latest_data, historical_data, context)
        all_observations.extend(nutrition_obs)
        logger.info(f"  - 营养指标: {len(nutrition_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 营养指标分析失败: {e}", exc_info=True)

    # 1.9 历史相似性
    try:
        dtw_obs = _observe_historical_similarity(latest_data, historical_data, context)
        all_observations.extend(dtw_obs)
        logger.info(f"  - 模式匹配: {len(dtw_obs)} 个观察点")
    except Exception as e:
        logger.error(f"❌ 模式匹配分析失败: {e}", exc_info=True)

    # 2. 排序观察结果
    all_observations.sort(key=lambda x: x['score_weight'], reverse=True)

    # 3. 计算关注度
    attention_level, level_desc, attention_score, color = _calculate_attention_score(all_observations)

    # 4. 生成摘要
    summary = _generate_summary_statement(attention_level, all_observations)

    logger.info(f"\n✅ 数据分析完成:")
    logger.info(f"  - 关注度级别: {attention_level}")
    logger.info(f"  - 关注度评分: {attention_score}")
    logger.info(f"  - 观察点数量: {len(all_observations)}")
    logger.info("="*60 + "\n")

    # 5. 返回结构化结果
    return {
        'attention_level': attention_level,
        'level_description': level_desc,
        'attention_score': attention_score,
        'color': color,
        'summary_statement': summary,
        'observations': all_observations,
        'observation_count': len(all_observations),
        'high_attention_count': len([o for o in all_observations if o.get('attention_level') == 'high']),
        'medium_attention_count': len([o for o in all_observations if o.get('attention_level') == 'medium']),
        'disclaimer': DISCLAIMER
    }
