# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# main.py

# --- 1. 标准库与第三方库导入 ---
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors
import matplotlib.colors as mcolors
import logging
import sys
import time
import traceback
import uuid

# --- 2. 自定义模块导入 ---
import config
import app_core
import app_state
import data_manager
import feature_engineering
import analysis_engine
import explainability
import controller
import risk_engine
import simulation_engine
import ocr_importer
import language_support


def setup_logging():
    """配置根日志记录器"""
    # 从 config 文件获取配置
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = getattr(config, 'LOG_FILE', 'health_monitor.log')

    # 定义日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )

    # 1. 配置日志到文件 (FileHandler)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)

    # 2. 配置日志到控制台 (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(log_level)

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 防止 Streamlit 重复运行导致处理器重复添加
    if not root_logger.hasHandlers():
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

    logging.info("日志系统已配置完成。")


def _get_original_lab_indicators() -> list:
    """
    从 config 中提取所有原始化验指标（排除衍生指标）
    
    :return: 原始指标名称列表
    """
    original_indicators = []
    
    # 从所有化验单模板中提取
    for template_name, items in config.LAB_REPORT_CONFIG.items():
        for item in items:
            indicator_name = item.get('name')
            if indicator_name:
                original_indicators.append(indicator_name)
    
    # 去重并排序
    return sorted(list(set(original_indicators)))


def create_baseline_gauge(indicator: str, z_score: float, z_threshold: float = 3.5) -> go.Figure:
    """
    创建基线偏离仪表盘（类似汽车速度表）
    
    :param indicator: 指标名称
    :param z_score: 修正Z-score（可为负）
    :param z_threshold: 高风险阈值（默认3.5）
    :return: Plotly Figure对象
    """
    abs_z = abs(z_score)
    
    # 确定颜色和状态
    if abs_z > z_threshold:
        color = '#e74c3c'  # 红色
        status = '显著偏离'
    elif abs_z > 2.5:
        color = '#f39c12'  # 橙色
        status = '中度偏离'
    elif abs_z > 1.5:
        color = '#3498db'  # 蓝色
        status = '轻度偏离'
    else:
        color = '#2ecc71'  # 绿色
        status = '正常范围'
    
    # 创建仪表盘
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs_z,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{indicator}</b><br><span style='font-size:0.8em'>{status}</span>",
            'font': {'size': 16}
        },
        number={'suffix': "σ", 'font': {'size': 24}},
        gauge={
            'axis': {
                'range': [None, max(5.0, abs_z * 1.2)],
                'tickwidth': 1,
                'tickcolor': "darkgray"
            },
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1.5], 'color': 'rgba(46, 204, 113, 0.2)'},   # 绿色区
                {'range': [1.5, 2.5], 'color': 'rgba(52, 152, 219, 0.2)'}, # 蓝色区
                {'range': [2.5, z_threshold], 'color': 'rgba(243, 156, 18, 0.2)'}, # 橙色区
                {'range': [z_threshold, max(5.0, abs_z * 1.2)], 'color': 'rgba(231, 76, 60, 0.2)'} # 红色区
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': z_threshold
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family='Arial, sans-serif', size=12)
    )
    
    return fig



def create_short_term_forecast_chart(
    historical_data: pd.DataFrame,
    indicator: str,
    mogp_result: dict,
    ref_range: tuple = (None, None),
    overall_last_date: datetime = None
) -> go.Figure:
    """
    创建短期预测图表（7-14天）
    
    特性：
    - 历史数据 + 预测数据
    - 置信区间（扇形填充）
    - 参考范围区域
    - 悬停显示详细信息
    - 动态配色（根据置信度）

    :param historical_data: 历史数据（通常为稳定期数据）
    :param indicator: 指标名称
    :param mogp_result: MOGP 预测结果
    :param ref_range: 参考范围
    :param overall_last_date: 全局最新检测日期（用于计算复查日期）
    :return: Plotly Figure对象
    """
    logger.debug(f"创建短期预测图表 - 指标: {indicator}")
    
    # === 1. 数据准备（统一日期格式）===
    hist_dates = historical_data.index
    hist_values = historical_data[indicator].values
    
    # 统一转换所有历史日期为 Python datetime
    hist_dates_native = []
    for d in hist_dates:
        if isinstance(d, pd.Timestamp):
            hist_dates_native.append(d.to_pydatetime())
        elif isinstance(d, datetime):
            hist_dates_native.append(d)
        else:
            # 处理任何其他类型（如 numpy.datetime64）
            hist_dates_native.append(pd.Timestamp(d).to_pydatetime())
    
    future_dates = mogp_result.get('future_dates', [])
    pred_mean = mogp_result.get('predicted_mean', [])
    pred_lower = mogp_result.get('confidence_lower', [])
    pred_upper = mogp_result.get('confidence_upper', [])
    confidence = mogp_result.get('confidence', 'medium')
    next_check_days = mogp_result.get('next_check_days', 60)

    # 限制预测范围
    max_forecast_days = min(14, len(future_dates))
    future_dates = future_dates[:max_forecast_days]
    pred_mean = pred_mean[:max_forecast_days]
    pred_lower = pred_lower[:max_forecast_days]
    pred_upper = pred_upper[:max_forecast_days]
    
    # 统一转换所有预测日期为 Python datetime
    future_dates_native = []
    for d in future_dates:
        if isinstance(d, pd.Timestamp):
            future_dates_native.append(d.to_pydatetime())
        elif isinstance(d, datetime):
            future_dates_native.append(d)
        elif isinstance(d, (int, float)):
            # 如果 future_dates 中混入了整数，转换为日期
            logger.warning(f"⚠️ 检测到预测日期为数值类型: {d}，尝试转换...")
            # 假设这是 "天数偏移"，从最后一个历史日期开始计算
            if hist_dates_native:
                base_date = hist_dates_native[-1]
                future_dates_native.append(base_date + timedelta(days=int(d)))
            else:
                logger.error(f"无法转换数值日期 {d}，跳过")
                continue
        else:
            # 其他类型统一用 pd.Timestamp 处理
            future_dates_native.append(pd.Timestamp(d).to_pydatetime())
    
    # === 2. 类型验证（防御性编程）===
    # 确保所有日期都是 datetime 类型
    all_dates = hist_dates_native + future_dates_native
    non_datetime = [d for d in all_dates if not isinstance(d, datetime)]
    if non_datetime:
        logger.error(f"❌ 检测到非 datetime 类型的日期: {non_datetime[:3]}...")
        st.error("图表数据格式错误，请联系技术支持")
        return go.Figure()  # 返回空图表
    
    # === 3. 创建图表 ===
    fig = go.Figure()
    
    # 历史数据
    fig.add_trace(go.Scatter(
        x=hist_dates_native,
        y=hist_values,
        mode='lines+markers',
        name='历史数据',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>实际值: %{y:.2f}<extra></extra>'
    ))
    
    # 预测均值
    confidence_map = {'high': '可靠', 'medium': '一般', 'low': '参考'}
    pred_color = {
        'high': '#2ecc71',
        'medium': '#f39c12',
        'low': '#e74c3c'
    }.get(confidence, '#95a5a6')
    
    fig.add_trace(go.Scatter(
        x=future_dates_native,
        y=pred_mean,
        mode='lines+markers',
        name=f'预测趋势 ({confidence_map[confidence]}置信)',
        line=dict(color=pred_color, width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>预测: %{y:.2f}<extra></extra>'
    ))
    
    # 置信区间
    fig.add_trace(go.Scatter(
        x=list(future_dates_native) + list(reversed(future_dates_native)),
        y=list(pred_upper) + list(reversed(pred_lower)),
        fill='toself',
        fillcolor=f'rgba({int(pred_color[1:3], 16)}, {int(pred_color[3:5], 16)}, {int(pred_color[5:7], 16)}, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='波动范围（95%）',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # 参考范围
    ref_lower, ref_upper = ref_range
    if ref_upper and pd.notna(ref_upper):
        fig.add_hrect(
            y0=ref_upper, y1=ref_upper * 1.5,
            fillcolor='rgba(231, 76, 60, 0.1)',
            line_width=0,
            annotation_text="高于正常",
            annotation_position="top right",
            annotation=dict(font=dict(size=10, color='red'))
        )
        fig.add_hline(
            y=ref_upper,
            line_dash="dot",
            line_color="rgba(231, 76, 60, 0.5)",
            annotation_text=f"正常上限: {ref_upper}",
            annotation_position="right",
            annotation=dict(font=dict(size=10))
        )
    
    if ref_lower and pd.notna(ref_lower):
        fig.add_hrect(
            y0=ref_lower * 0.5, y1=ref_lower,
            fillcolor='rgba(52, 152, 219, 0.1)',
            line_width=0,
            annotation_text="偏低区域",
            annotation_position="bottom right",
            annotation=dict(font=dict(size=10, color='blue'))
        )
        fig.add_hline(
            y=ref_lower,
            line_dash="dot",
            line_color="rgba(52, 152, 219, 0.5)",
            annotation_text=f"正常下限: {ref_lower}",
            annotation_position="right",
            annotation=dict(font=dict(size=10))
        )
    
    # --- 复查时间提示标注 ---
    if next_check_days and pd.notna(next_check_days):
        # 确定计算基准日期
        if overall_last_date:
            base_date_for_annotation = overall_last_date
            source_label = "全局最新检测日期"
        elif hist_dates_native:  # 如果没有传入，回退到历史数据
            base_date_for_annotation = hist_dates_native[-1]
            source_label = "稳定期最新日期"
            logger.warning("⚠️ 未传入 overall_last_date，使用稳定期数据的最后日期")
        else:
            logger.warning("⚠️ 无可用日期，跳过复查标注")
            base_date_for_annotation = None
        
        if base_date_for_annotation:
            try:
                # 统一类型转换
                if isinstance(base_date_for_annotation, pd.Timestamp):
                    base_date_dt = base_date_for_annotation.to_pydatetime()
                elif isinstance(base_date_for_annotation, datetime):
                    base_date_dt = base_date_for_annotation
                else:
                    base_date_dt = pd.Timestamp(base_date_for_annotation).to_pydatetime()
                
                # 类型最终验证
                if not isinstance(base_date_dt, datetime):
                    raise TypeError(f"类型转换失败: {type(base_date_dt)}")
                
                # 计算复查日期
                days_to_add = int(float(next_check_days))
                next_check_date = base_date_dt + timedelta(days=days_to_add)
                next_check_date_timestamp_ms = next_check_date.timestamp() * 1000
                
                # 添加标注
                fig.add_vline(
                    x=next_check_date_timestamp_ms,
                    line_dash="dot",
                    line_color="cyan",
                    annotation_text=f"复查时间提示 ({days_to_add}天后)",
                    annotation_position="top"
                )
                
                logger.debug(
                    f"✅ 复查时间提示标注成功 (基于{source_label} "
                    f"{base_date_dt.strftime('%Y-%m-%d')}): "
                    f"{next_check_date.strftime('%Y-%m-%d')}"
                )
            
            except Exception as e:
                logger.error(
                    f"❌ 复查日期标注失败: {e}\n"
                    f"  base_date type: {type(base_date_for_annotation)}\n"
                    f"  source: {source_label}",
                    exc_info=True
                )
    
    elif not next_check_days or pd.isna(next_check_days):
        logger.debug("未设置复查日期（next_check_days 为空或 NaN）")
    elif len(hist_dates_native) == 0:
        logger.debug("无历史日期数据，无法计算复查日期")
    
    # === 4. 布局美化 ===
    fig.update_layout(
        title=dict(
            text=f"<b>{indicator}</b> - 短期趋势预测（未来 {max_forecast_days} 天）",
            font=dict(size=18, color='#ecf0f1'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(52, 73, 94, 0.4)',
            zeroline=False,
            color='#bdc3c7'
        ),
        yaxis=dict(
            title=f'{indicator} 数值',
            showgrid=True,
            gridcolor='rgba(52, 73, 94, 0.4)',
            zeroline=False,
            color='#bdc3c7'
        ),
        hovermode='x unified',
        plot_bgcolor='#2c3e50',
        paper_bgcolor='#34495e',
        font=dict(family='Arial, sans-serif', size=12, color='#ecf0f1'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(44, 62, 80, 0.8)',
            bordercolor='#7f8c8d',
            borderwidth=1
        ),
        height=480,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig




def render_unified_feedback_module(observation_result: dict, patient_id: int):
    """
    统一反馈界面 - 合并常规反馈和历史回顾 - 支持所有警报类型的 SHAP 归因
    
    改进：
    - 为所有警报类型（Z-Score、模型、启发式）生成 SHAP 归因
    - 统一保存到 feedback_with_shap 表
    - 确保 analysis_engine 能从所有反馈中学习
    - 将两个独立卡片合并为一个统一的列表
    - 区分真实归因（real）和代理归因（proxy）
    - 清晰区分"新发现"和"历史相似"
    - 保持横向按钮布局
    """
    logger = logging.getLogger(__name__)
    logger.debug("渲染统一反馈模块 (V8.0 - 归因类型版)...")
    
    # --- 步骤 1：收集所有需要反馈的项目 ---
    all_labels = observation_result.get('context', {}).get('all_labels', {})
    all_observations = observation_result.get('observations', [])
    
    # 收集常规反馈请求（AI 发现的异常）
    actionable_obs = [
        o for o in all_observations 
        if o.get('attention_level') in ['high', 'medium']
    ]
    
    regular_requests = []
    for obs in actionable_obs:
        obs_uuid = obs.get('report_uuid')
        if obs_uuid and obs_uuid not in all_labels:
            regular_requests.append({
                'uuid': obs_uuid,
                'type': 'regular',
                'indicator': obs.get('indicator', '综合'),
                'attention_level': obs.get('attention_level'),
                'observation': obs.get('observation', ''),
                'obs_ref': obs 
            })
    
    # 收集历史相似模式请求
    similarity_requests = []
    for obs in all_observations:
        if 'unified_feedback_request' in obs:
            req = obs['unified_feedback_request']
            similarity_requests.append({
                'uuid': req['uuid'],
                'type': 'similarity',
                'indicator': req['indicator'],
                'date_str': req['date_str'],
                'similarity_pct': req['similarity_pct'],
                'matched_date_str': req['matched_date_str'],
                'matched_date_obj': req['matched_date_obj'],
                'obs_ref': obs
            })
    
    # 去重（使用 uuid + indicator 组合键）
    seen_keys = set()
    all_requests = []
    
    for req in regular_requests:
        key = (req['uuid'], req['indicator'])
        if key not in seen_keys:
            all_requests.append(req)
            seen_keys.add(key)
    
    for req in similarity_requests:
        key = (req['uuid'], req['indicator'])
        if key not in seen_keys:
            all_requests.append(req)
            seen_keys.add(key)
    
    # --- 步骤 2：初始化已处理列表（会话级别持久化）---
    session_key = f"processed_unified_requests_{patient_id}"
    if session_key not in st.session_state:
        st.session_state[session_key] = set()
    
    # 过滤已处理的
    pending_requests = [
        req for req in all_requests 
        if (req['uuid'], req['indicator']) not in st.session_state[session_key]
    ]
    
    if not pending_requests:
        logger.debug("所有请求已处理，跳过渲染")
        return
    
    # --- 步骤 3：渲染统一卡片 ---
    st.markdown("---")
    st.markdown("### 💭 帮助 AI 理解您的历史模式")

    with st.expander("💡 为什么需要您的反馈？点击查看说明", expanded=False):
        st.markdown("""
        ### 为什么 AI 需要向您请教？
        
        **AI 发现了"异常"，但它不确定**：
        - 这是一个**重要的变化**（比如治疗效果不佳、病情波动）
        - 还是一个**正常的小波动**（比如感冒、劳累、测量误差）
        
        **举个例子**：
        
        假设您的某个指标在 2023年3月 突然升高了，AI 会想：
        - 🤔 "这个升高是因为那时正在化疗，属于正常反应吗？"
        - 🤔 "还是说这是病情加重的信号，需要引起警惕？"
        
        只有**您**最清楚当时的真实情况！
        
        ---
        
        ### 您的反馈能帮 AI 学到什么？
        
        **✅ 良性波动**：
        - AI 学到："原来这种情况是正常的，下次遇到不用太紧张"
        - 以后类似的小波动，AI 就不会误报警告
        
        **⚠️ 重要变化**：
        - AI 学到："这种情况需要重点关注，下次要提高警惕"
        - 以后遇到相似模式，AI 会主动提醒您检查
        
        **❌ 数据错误**：
        - AI 学到："这个数据不可信，要忽略它"
        - 避免错误数据干扰未来的判断
        
        ---
        
        ### 您的反馈完全保密
        
        - ✅ 只存储在您自己的电脑上
        - ✅ 不会上传到任何服务器
        - ✅ 只有您自己能看到
        
        ---
        
        ### 不确定如何标记？
        
        **没关系！您可以**：
        1. 点击「🙈 暂时忽略」，稍后再决定
        2. 咨询医生后再回来标记
        3. 标记错了也能修改（在「📂 历史记录」中）
        
        **💡 小提示**：即使只标记一两个，也能显著提升 AI 的准确性！
        """)

    st.info(
        "AI 发现以下日期的检测结果与历史模式不同，"
        "请告诉 AI 当时的情况："
    )
    
    # 批量忽略按钮
    if len(pending_requests) > 1:
        col_ignore, col_spacer = st.columns([2, 3])
        
        with col_ignore:
            if st.button(
                "🙈 暂时忽略所有请求", 
                use_container_width=True,
                help="稍后再决定，这些请求会在下次刷新时消失",
                key="ignore_all_unified"
            ):
                logger.info(f"用户批量忽略 {len(pending_requests)} 个统一请求")
                for req in pending_requests:
                    st.session_state[session_key].add((req['uuid'], req['indicator']))
                
                st.toast(f"✅ 已忽略 {len(pending_requests)} 个请求", icon="🙈")
                st.rerun()
        
        with col_spacer:
            st.caption(f"💡 或逐个确认下方 {len(pending_requests)} 个日期")
    
    st.markdown("---")
    
    # --- 步骤 4：逐个渲染请求---
    try:
        patient_data_raw = st.session_state.get('patient_data_raw')
        if patient_data_raw is None or patient_data_raw.empty:
            logger.warning("无法获取患者数据，跳过反馈渲染")
            return
        
        uuid_date_map = patient_data_raw[['report_uuid', 'phase']].copy()
        uuid_date_map = uuid_date_map.drop_duplicates(subset=['report_uuid'])
        uuid_date_map['report_date_str'] = uuid_date_map.index.strftime('%Y年%m月%d日')
        uuid_date_map = uuid_date_map.set_index('report_uuid')
        
    except Exception as e:
        logger.error(f"构建日期映射失败: {e}", exc_info=True)
        st.error("无法加载日期信息")
        return
    
    # --- 步骤 5：渲染每个请求（3个按钮的处理逻辑）---
    for req in pending_requests:
        req_uuid = req['uuid']
        req_type = req['type']
        req_indicator = req['indicator']
        
        # 验证 UUID 存在性
        if req_uuid not in uuid_date_map.index:
            logger.warning(f"UUID {req_uuid} 未在映射表中找到，跳过")
            continue
        
        # 提取日期和阶段
        date_str = uuid_date_map.loc[req_uuid, 'report_date_str']
        phase_str = uuid_date_map.loc[req_uuid, 'phase']
        
        # 构建唯一 key
        button_key = f"unified_{req_uuid}_{req_indicator.replace(' ', '_')}"
    
        # 横向布局：[日期+说明] [按钮1] [按钮2] [按钮3]
        col_date, col_btn1, col_btn2, col_btn3 = st.columns([3, 1, 1, 1])
        
        with col_date:
            # 根据类型显示不同的标题和说明
            if req_type == 'regular':
                st.markdown(f"**📅 {date_str}** ({phase_str})")
                indicator = req.get('indicator', '综合')
                level = req.get('attention_level', 'medium')
                level_emoji = {
                    'high': '🔴', 
                    'medium': '🟡', 
                    'low': '🟢'
                }.get(level, '⚪')

                st.caption(f"{level_emoji} {indicator}")
            
            elif req_type == 'similarity':
                st.markdown(f"**📅 {date_str}** ({req['indicator']})")
                st.caption(
                    f"💡 与近期相似度: **{req['similarity_pct']}%** "
                    f"（匹配到 {req['matched_date_str']} 的模式）"
                )
        
        # === 按钮处理逻辑（统一的 SHAP + shap_type 处理）===
        # 定义通用的保存函数
        def save_feedback(label: str, button_key_suffix: str):
            """统一的反馈保存逻辑（带 shap_type）"""
            try:
                logger.info(f"用户标记 ({req_uuid}, {req_indicator}) 为 '{label}'")
                
                # === 1. 获取观察项引用 ===
                obs = req.get('obs_ref')
                
                if obs is None:
                    logger.error(
                        f"无法获取观察项引用（UUID={req_uuid}, "
                        f"Indicator={req_indicator}）"
                    )
                    st.error("❌ 内部错误：观察项丢失")
                    return  # 提前返回，避免 continue 错误
                
                # === 2. 判断 SHAP 来源并确定归因类型 ===
                if 'shap_values' in obs and obs['shap_values']:
                    # 场景 A：有真实的 SHAP 值（来自模型警报）
                    shap_to_save = obs['shap_values']
                    observation_id = obs.get('id', str(uuid.uuid4()))
                    pattern_type = obs.get('pattern_type', 'model_anomaly')
                    
                    # 提取 shap_type（带验证）
                    shap_type = obs.get('shap_type', 'proxy')
                    logger.debug(f"  提取的原始 shap_type: {shap_type}")
                    
                    # 类型校验（防御性编程）
                    if shap_type not in ['real', 'proxy']:
                        logger.warning(
                            f"⚠️ 无效的 shap_type: {shap_type}，强制修正为 'proxy'"
                        )
                        shap_type = 'proxy'
                    
                    # 一致性检查
                    if shap_type == 'real' and len(shap_to_save) == 1:
                        logger.warning(
                            f"⚠️ 标记为 'real' 但只有单特征归因 ({list(shap_to_save.keys())}), "
                            f"自动修正为 'proxy'"
                        )
                        shap_type = 'proxy'

                    logger.info(f"  最终确定的 shap_type: {shap_type}") 

                else:
                    # 场景 B：无 SHAP 值（来自 Z-Score 或启发式警报）
                    # 创建代理归因：将 100% 责任归给当前指标
                    proxy_indicator = obs.get('indicator', 'Unknown_Feature')
                    shap_to_save = {proxy_indicator: 1.0}
                    observation_id = obs.get('id', str(uuid.uuid4()))
                    pattern_type = obs.get('pattern_type', 'z_score')
                    shap_type = 'proxy'  # 代理归因
                    
                    logger.info(
                        f"  创建代理 SHAP 归因：{proxy_indicator} = 1.0 "
                        f"(shap_type=proxy)\n"
                        f"  原因：此警报来自 {pattern_type} "
                        f"（无原生 SHAP 值）"
                    )
                
                # === 3. 统一保存反馈（新增 shap_type）===
                data_manager.save_feedback_with_shap(
                    patient_id=patient_id,
                    report_uuid=req_uuid,
                    indicator=req_indicator,
                    label=label,
                    shap_values_dict=shap_to_save,
                    observation_uuid=observation_id,
                    pattern_type=pattern_type,
                    shap_type=shap_type
                )
                logger.info(
                    f"  ✅ 反馈已保存到 feedback_with_shap 表 "
                    f"(shap_type={shap_type})"
                )
                
                # === 4. 相似度反馈）===
                if req_type == 'similarity':
                    is_similar = (label != 'lab_error')  # 数据错误时相似度无效
                    data_manager.save_similarity_feedback(
                        current_uuid=req_uuid,
                        indicator=req['indicator'],
                        is_similar=is_similar
                    )
                
                # === 5. 标记为已处理 ===
                current_key = (req_uuid, req_indicator)
                st.session_state[session_key].add(current_key)
                st.session_state.labels_changed = True
                
                # 调试日志
                logger.info(
                    f"✅ 已标记 ({req_uuid[:8]}..., {req_indicator}) 为已处理\n"
                    f"  当前已处理总数: {len(st.session_state[session_key])}"
                )
                
                # === 6. 显示反馈（根据标签类型）===
                label_messages = {
                    'benign': ("✅", "良性波动"),
                    'significant': ("⚠️", "重要变化"),
                    'lab_error': ("❌", "数据错误")
                }
                emoji, label_cn = label_messages[label]
                
                message = f"{emoji} 已将 {date_str} 的 {req_indicator} 标记为“{label_cn}”"
                
                if req_type == 'similarity':
                    if label == 'lab_error':
                        message += (
                            f"\n\n**AI 学到了**：\n"
                            f"1️⃣ 那次数据不可信（将从训练集中排除）\n"
                            f"2️⃣ 与 {req['matched_date_str']} 的相似度匹配无效"
                        )
                    else:
                        action_text = "不需要担心" if label == 'benign' else "值得重点关注"
                        message += (
                            f"\n\n**AI 学到了**：\n"
                            f"1️⃣ 与 {req['matched_date_str']} 的相似度判断是准确的\n"
                            f"2️⃣ 那次事件{action_text}"
                        )
                
                if label == 'lab_error':
                    message += "\n\n💡 提示：修正数据后重新训练模型"
                
                # 显示消息
                placeholder = st.empty()
                if label == 'benign':
                    placeholder.success(message)
                elif label == 'significant':
                    placeholder.warning(message)
                else:
                    placeholder.info(message)
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"标记失败: {e}", exc_info=True)
                st.error(f"❌ 标记失败: {e}")
        
        # 按钮 1: 良性波动
        with col_btn1:
            if st.button(
                "✅ 良性波动", 
                key=f"{button_key}_benign",
                use_container_width=True,
                help="当时是感冒、劳累等正常波动"
            ):
                save_feedback('benign', 'benign')

        # 按钮 2: ⚠️ 重要变化
        with col_btn2:
            if st.button(
                "⚠️ 重要变化", 
                key=f"{button_key}_significant",
                use_container_width=True,
                help="需要临床关注"
            ):
                save_feedback('significant', 'significant')

        # 按钮 3: 数据错误
        with col_btn3:
            if st.button(
                "❌ 数据错误", 
                key=f"{button_key}_lab_error",
                use_container_width=True,
                help="录入有误或检测异常"
            ):
                save_feedback('lab_error', 'lab_error')

        
def render_mogp_core_metrics(
    indicators_with_prediction: list,
    mogp_results_cache: dict,
    patient_data_raw: pd.DataFrame,
    mogp_last_updated: datetime
):
    """
    【新增函数】渲染 MOGP 核心指标卡片（3 列布局）
    
    :param indicators_with_prediction: 有预测结果的指标列表
    :param mogp_results_cache: MOGP 结果缓存
    :param patient_data_raw: 原始患者数据
    :param mogp_last_updated: MOGP 最后更新时间
    """
    logger = logging.getLogger(__name__)
    
    # 提取第一个指标的结果（用于统计）
    first_indicator = indicators_with_prediction[0]
    first_result = mogp_results_cache[first_indicator]
    actual_mogp_confidence = first_result.get('confidence', 'medium')
    
    # 计算数据统计
    if patient_data_raw is not None and not patient_data_raw.empty:
        time_span = (patient_data_raw.index.max() - patient_data_raw.index.min()).days
        total_records = len(patient_data_raw)
        avg_interval = time_span / max(1, total_records - 1)
    else:
        time_span = 0
        total_records = 0
        avg_interval = 0
    
    # 显示状态卡片
    st.info(f"✨ **正在观察 {len(indicators_with_prediction)} 个关键指标的变化趋势（未来7-14天）**")
    st.caption(f"包括：{', '.join(indicators_with_prediction[:3])}{'等' if len(indicators_with_prediction) > 3 else ''}")
    
    # === 核心指标展示（3列布局）===
    col1, col2, col3 = st.columns(3)

    with col1:
        next_check = first_result.get('next_check_days', 60)
        st.metric(
            "复查间隔提示", 
            f"约 {next_check} 天后", 
            delta="根据检测间隔计算",
            delta_color="off",
            help=(
                f"**基于您的历史检测频率自动计算**\n\n"
                f"📊 数据统计：\n"
                f"- 平均检测间隔: {avg_interval:.0f}天/次\n"
                f"- 数据跨度: {time_span/30:.1f}个月\n"
                f"- 总记录数: {total_records} 次"
            )
        )

    with col2:
        pred_window = len(first_result.get('future_dates', []))
        st.metric(
            "预测窗口", 
            f"{pred_window} 天", 
            help="根据您的检测频率自动调整，最长14天" 
        )

    with col3:
        # 预测可靠性（唯一的核心指标）
        confidence_map = {
            'high': (
                '🟢 可靠', 
                f"**综合评估：可靠**\n\n"
                f"✅ 数据充足\n"
                f"✅ 模型拟合良好\n"
                f"✅ 预测不确定性低\n\n"
                f"**如何使用**：\n"
                f"- 趋势方向（上升/下降）：高度可信\n"
                f"- 具体数值：可作为重要参考\n"
                f"- 提示：可把趋势观察结果带给医生讨论"
            ),
            'medium': (
                '🟡 一般', 
                f"**综合评估：一般**\n\n"
                f"⚠️ 数据存在波动或模型拟合有限\n"
                f"⚠️ 预测置信区间较宽\n\n"
                f"**如何使用**：\n"
                f"- 趋势方向（上升/下降）：可信\n"
                f"- 具体数值：仅供参考（预测范围会较宽）\n"
                f"- 提示：重点观察趋势，不要把单一预测数值作为决策依据\n\n"
                f"💡 这是正常现象：患者的指标本身就波动大，\n"
                f"模型「老实」地表达了不确定性"
            ),
            'low': (
                '🔴 参考', 
                f"**综合评估：参考**\n\n"
                f"⚠️ 数据有限或数据波动较大\n"
                f"⚠️ 模型拟合欠佳或不确定性高\n\n"
                f"**当前限制**：\n"
                f"- 趋势方向：仅供参考\n"
                f"- 具体数值：不应用作决策依据\n"
                f"- 提示：优先关注实际检测结果，并咨询医生\n\n"
                f"**如何改善**：\n"
                f"1. 增加检测频率（如治疗期每月一次）\n"
                f"2. 积累更多数据（至少8-10次）\n"
                f"3. 排查数据是否有录入错误\n\n"
                f"💡 点击下方「如何改善预测质量？」查看详细提示"
            )
        }
        
        confidence_display, confidence_help = confidence_map.get(
            actual_mogp_confidence,
            ('🟡 一般', '模型评估中...')
        )
        
        st.metric(
            "预测可靠性",
            confidence_display, 
            delta=None,  # ✅ 删除 delta
            delta_color="off",
            help=confidence_help
        )
    
    # 警告提示（如果有）
    warning = first_result.get('warning')
    if warning:
        warning_map = {
            '⚠️ 不确定性快速增长，预测仅供参考': '⚠️ 数据波动较大，预测结果仅供参考，请结合医生判断',
            'ℹ️ 历史拟合误差相对较大，请结合基线观察': 'ℹ️ 数据拟合度一般，请结合个人基线观察'
        }
        warning_cn = warning_map.get(warning, warning)
        st.warning(f"💡 {warning_cn}")
    
    # 时间戳
    if mogp_last_updated:
        update_time_str = mogp_last_updated.strftime('%Y-%m-%d %H:%M')
        st.caption(f"预测生成时间: {update_time_str}")




def render_mogp_status_card(diagnostic_info: dict, has_prediction: bool):
    """
    【V5 - 极简版】只保留预测可靠性
    """
    logger = logging.getLogger(__name__)
    
    if not diagnostic_info:
        logger.warning("MOGP 状态卡：未收到 diagnostic_info，无法渲染。")
        return

    selected = diagnostic_info.get('selected', [])
    total_records = diagnostic_info.get('total_records', 0)
    
    # === 场景 1: 成功（有预测结果）===
    if has_prediction and selected:
        st.info(f"✨ **正在观察 {len(selected)} 个关键指标的变化趋势（未来7-14天）**")
        st.caption(f"包括：{', '.join(selected[:3])}{'等' if len(selected) > 3 else ''}")
        
        # 获取 MOGP 的实际可靠性
        actual_confidence = st.session_state.get('actual_mogp_confidence', 'medium')
        logger.info(f"render_mogp_status_card 收到的实际可靠性: {actual_confidence}")
        
        # 获取数据统计（用于详细说明）
        patient_data_raw = st.session_state.get('patient_data_raw')
        
        if patient_data_raw is not None and not patient_data_raw.empty and total_records > 0:
            time_span = (patient_data_raw.index.max() - patient_data_raw.index.min()).days
            avg_interval = time_span / max(1, total_records - 1)
        else:
            time_span = 0
            avg_interval = 0
        
        # ✅ 只在可靠性为 'low' 时展开详细说明
        if actual_confidence == 'low':
            with st.expander("💡 如何改善预测质量？", expanded=False):
                st.markdown(f"""
                **当前状态**：
                - 记录数：{total_records} 次
                - 平均间隔：{avg_interval:.0f} 天/次
                - 数据跨度：{time_span/30:.1f} 个月
                
                **为什么可靠性显示为"参考"？**
                
                系统综合评估后发现：
                - ⚠️ 预测不确定性较大（模型的"自信度"不足）
                - ⚠️ 历史拟合误差较大（可能存在数据波动或异常值）
                
                **改善提示**：
                1. **增加检测频率**：如医生认为有必要，可在关键时期增加检测频率
                2. **积累更多数据**：至少需要 8-10 次稳定期检测
                3. **排查异常值**：检查历史数据是否有录入错误或实验室误差
                
                **当前提示**：
                - 趋势方向：仅供参考
                - 具体数值：不应用作决策依据
                - 优先关注实际检测结果，预测仅作为趋势观察
                """)
                
                # 如果检测频率特别低，额外提示
                if avg_interval > 90:
                    st.warning(
                        f"💡 **温馨提示**：您的平均检测间隔为 {avg_interval:.0f} 天（约{avg_interval/30:.1f}个月）。\n\n"
                        f"虽然这对于长期监控是合理的，但如医生认为有必要，可在关键治疗期适当增加检测频率。"
                        f"是否调整复查频率请咨询医生。"
                    )
        
    # === 场景 2: 首次引导（总数据 < 5）===
    elif total_records < 5:
        needed = 5 - total_records
        st.info("📊 **正在建立您的健康档案**")
        
        st.progress(total_records / 5, text=f"数据积累进度: {total_records} / 5")
        st.caption(f"再记录 **{needed}** 次检查，就能看到您的专属趋势预测了！")
    
    # === 场景 3: 积累期（总数据 >= 5，但具体指标不足）===
    else:
        st.info("📊 **数据积累中... 趋势预测即将开启！**")
        
        candidates = diagnostic_info.get('candidates', {})
        best_candidate_name = None
        max_count = 0
        
        if candidates:
            for indicator, info in candidates.items():
                if not info.get('sufficient', False) and info.get('count', 0) > max_count:
                    max_count = info['count']
                    best_candidate_name = indicator
        
        if best_candidate_name and max_count > 0:
            needed = 5 - max_count
            st.progress(max_count / 5, text=f"最接近的指标: {max_count} / 5")
            st.caption(f"例如：**{best_candidate_name}** 已有 {max_count} 次, 再录入 **{needed}** 次即可解锁该指标的预测。")
        else:
            st.caption("请继续录入肿瘤标志物或血常规数据，系统将在数据充足后自动开始预测。")


# --- 3. Streamlit 页面配置与初始化 ---
st.set_page_config(layout="wide", page_title="CareTrace 关照轨迹")
# 配置日志系统
setup_logging()

# 为 main.py 自身获取一个日志记录器
logger = logging.getLogger(__name__)

logger.info("=======================================")
logger.info("Streamlit 脚本开始执行 (Rerun)")
logger.info("=======================================")


language_support.init_language()
language_support.language_selector()

st.title(language_support.t("app_title"))
st.markdown("---")

# 初始化数据库
try:
    data_manager.init_db()
    logger.debug("数据库初始化 (init_db) 调用完成。")
except Exception as e:
    logger.error(f"数据库初始化失败: {e}", exc_info=True)
    st.error(f"数据库连接失败: {e}")
    st.stop() # 如果数据库失败，停止应用


# --- 4. 侧边栏:病人管理 ---
st.sidebar.header(language_support.t("patient_profile"))
logger.debug("渲染侧边栏...")
patients_df = data_manager.get_patients()
select_patient_placeholder = language_support.t("select_patient_placeholder")
patient_list = [select_patient_placeholder] + patients_df['name'].tolist()

default_index = 0
if len(patients_df) == 1:
    default_index = 1

selected_patient_name = st.sidebar.selectbox(
    language_support.t("select_patient"),
    patient_list,
    index=default_index,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
with st.sidebar.expander(language_support.t("new_profile")):
    new_patient_name = st.text_input(language_support.t("enter_name"), label_visibility="collapsed")
    if st.button(language_support.t("create_profile"), width='stretch'):
        logger.info(f"用户点击 '创建档案'，姓名: {new_patient_name}")
        if new_patient_name:
            if data_manager.add_patient(new_patient_name):
                logger.info(f"档案 '{new_patient_name}' 创建成功。")
                st.success(language_support.t("created"))
                st.rerun()
            else:
                logger.warning(f"档案 '{new_patient_name}' 创建失败：档案已存在。")
                st.error(language_support.t("profile_exists"))
        else:
            logger.warning("用户尝试创建档案但未输入姓名。")
            st.warning(language_support.t("name_required"))

# --- 5. 主界面核心逻辑 ---
if selected_patient_name != select_patient_placeholder:
    patient_id = int(patients_df[patients_df['name'] == selected_patient_name]['id'].iloc[0])
    logger.info(f"病人已选择: {selected_patient_name} (ID: {patient_id})")

    # 使用 controller
    @st.cache_resource
    def get_controller(p_id):
        logger.info(f"******* 正在创建新的 HealthController (PID: {p_id}) (应仅在首次加载或数据更改后出现) *******")
        return controller.HealthController(p_id, state=app_state.StreamlitStateStore())

    health_controller = get_controller(patient_id)
    logger.debug(f"HealthController (PID: {patient_id}) 已从缓存中获取。")

    # 初始化页面状态。核心逻辑通过状态适配器访问，不直接依赖 Streamlit。
    state_store = app_state.StreamlitStateStore()
    app_state.initialize_app_state(state_store)
    
    # 使用 controller 加载数据
    logger.debug("调用 health_controller.get_processed_data()...")
    patient_data_raw, patient_data, ref_ranges_dict = health_controller.get_processed_data()
    logger.debug(f"数据加载完成。Raw: {patient_data_raw.shape}, Processed: {patient_data.shape}")

    # MOGP 恢复逻辑移到核心工作流，页面只根据结果决定是否刷新展示数据。
    mogp_restore = app_core.restore_mogp_for_patient(
        health_controller,
        patient_id,
        patient_data_raw,
        state_store
    )
    logger.info(f"MOGP 恢复状态: {mogp_restore['message']}")
    if mogp_restore.get('force_refresh'):
        logger.info("正在将恢复的 MOGP 特征合并到患者数据...")
        patient_data_raw, patient_data, ref_ranges_dict = health_controller.get_processed_data(
            force_refresh=True
        )
        
    # 提取特征列
    if not patient_data.empty:
        numeric_cols = patient_data.select_dtypes(include=np.number).columns
        all_feature_columns = [
            col for col in numeric_cols 
            if col not in ['report_uuid']
        ]
    else:
        all_feature_columns = []

    # --- 创建选项卡 ---
    dashboard, data_entry, data_management, simulation_tab = st.tabs([
        language_support.t("tab_report"),
        language_support.t("tab_entry"),
        language_support.t("tab_history"),
        language_support.t("tab_simulation"),
    ])


    # --- Tab 1: 健康报告 ---
    with dashboard:
        patient_id = health_controller.patient_id
        logger.debug("渲染 '健康报告' 选项卡...")

        # --- 0. 免责声明 ---
        st.warning(language_support.t("medical_disclaimer"))
        # --- 1. 前置检查与数据加载 ---
        patient_data_raw, patient_data, ref_ranges_dict = health_controller.get_processed_data(
            force_refresh=st.session_state.data_changed
        ) # 如果数据变了，强制刷新
        
        # 将 patient_data_raw 存入 session_state
        st.session_state.patient_data_raw = patient_data_raw 

        if patient_data.empty:
            st.info(language_support.t("no_health_data"))
        else:
            # --- 2. 显示最近检测信息 ---
            try:
                latest_row = patient_data.iloc[-1]
                latest_date = latest_row.name
                latest_phase = latest_row['phase']
                date_format = "%Y年%m月%d日" if language_support.get_language() == "zh" else "%Y-%m-%d"
                st.markdown(language_support.t(
                    "latest_test",
                    date=latest_date.strftime(date_format),
                    phase=language_support.phase_label(latest_phase),
                ))
            except (IndexError, KeyError):
                st.warning(language_support.t("latest_missing"))
                latest_date = None # 标记一下，后面会用到

            st.markdown("---")

            # --- 3. 核心：获取评估结果 ---
            #    每次访问仪表盘都重新评估风险（因为基线参数可能已校准）
            with st.spinner(language_support.t("analyzing_data")):
                try:
                    observation_result = app_core.assess_current_risk(health_controller)
                    # 评估后，数据状态变为“未更改”（直到下次录入）
                    st.session_state.data_changed = False
                except Exception as assess_e:
                    logger.error(f"调用 assess_current_risk 失败: {assess_e}", exc_info=True)
                    st.error(language_support.t("report_error", error=assess_e))
                    observation_result = {'attention_level': 'unknown', 'message': '评估失败', 'observations': []}

            if observation_result.get('attention_level') == 'unknown':
                st.warning(observation_result.get('message', '无法生成评估报告'))
            else:
                # =========================================
                # 🎯 金字塔结构：警报 > 趋势 > 细节 > 反馈
                # =========================================

                # --- 第1层：高优先级警报（基线偏离）---
                baseline_alerts = [
                    obs for obs in observation_result.get('observations', [])
                    if obs.get('pattern_type') == 'baseline_deviation'
                    and obs.get('attention_level') in ['high', 'medium']
                ]

                if baseline_alerts:
                    st.error(language_support.t("baseline_deviation"))
                    logger.info(f"UI: 渲染 {len(baseline_alerts)} 个基线偏离警报")

                    # 为每个警报创建一个可展开的区域
                    for alert in baseline_alerts:
                        with st.expander(
                            f"⚠️ {alert['indicator']} - {alert.get('severity', '异常')} ({alert.get('attention_level', '').capitalize()})",
                            expanded=(alert['attention_level'] == 'high') # 高风险默认展开
                        ):
                            col1, col2 = st.columns([1, 2]) # 左窄右宽

                            with col1:
                                # 左侧：仪表盘
                                try:
                                    gauge_fig = create_baseline_gauge(
                                        alert['indicator'],
                                        alert.get('z_score', 0),
                                        # 使用校准后的阈值
                                        z_threshold=health_controller.baseline_monitor.z_thresholds['high']
                                    )
                                    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
                                except Exception as gauge_e:
                                    logger.error(f"渲染仪表盘失败 ({alert['indicator']}): {gauge_e}", exc_info=True)
                                    st.error(language_support.t("gauge_failed"))

                            with col2:
                                # 右侧：详细信息
                                st.markdown(language_support.t(
                                    "current_value",
                                    value=f"{alert.get('current_value', 'N/A'):.2f}",
                                ))
                                st.markdown(language_support.t(
                                    "personal_baseline",
                                    value=f"{alert.get('baseline_median', 'N/A'):.2f}",
                                ))
                                
                                # 将 Z-score 转化为用户语言
                                z_score = alert.get('z_score', 0)
                                deviation_level = "严重偏离" if abs(z_score) > 3.5 else \
                                                "中度偏离" if abs(z_score) > 2.5 else \
                                                "轻度偏离" if abs(z_score) > 1.5 else "正常范围"
                                
                                st.markdown(language_support.t(
                                    "deviation_degree",
                                    level=language_support.deviation_label(deviation_level),
                                ))
                                st.caption(language_support.t("z_caption", z_score=z_score))

                                # 显示与医学参考范围的关系
                                context_list = alert.get('clinical_context', [])
                                if context_list:
                                    for ctx in context_list:
                                        st.caption(f"🩺 {ctx}") # 用 caption 显示

                                st.markdown("---")
                                st.markdown(language_support.t("interpretation"))
                                # 使用 markdown 并允许 HTML (用于加粗等)
                                st.markdown(
                                    alert.get('interpretation', language_support.t("no_interpretation")),
                                    unsafe_allow_html=True,
                                )

                                st.markdown(language_support.t("observation_note"))
                                st.warning(alert.get('note', language_support.t("no_observation_note")))

                    st.markdown("---") # 在所有警报下方加分隔线

                # --- 第2层：短期趋势预测（MOGP）---
                st.markdown(language_support.t("short_term_prediction"))
                logger.debug("UI: 准备渲染短期趋势预测...")

                # 尝试从 session_state 加载 MOGP 结果
                mogp_results_cache = st.session_state.get('mogp_results') or {}
                mogp_last_updated = st.session_state.get('mogp_last_updated')
                mogp_diagnostic = st.session_state.get('mogp_diagnostic_info', {})

                # 健壮性检查：记录日志
                if not mogp_results_cache:
                    logger.warning("⚠️ session_state.mogp_results 为空或 None，无法渲染预测图表")
                else:
                    logger.debug(f"✅ 成功获取 MOGP 缓存，共 {len(mogp_results_cache)} 个指标")

                # 检查是否有有效的 MOGP 结果
                indicators_with_prediction = [
                    ind for ind, res in mogp_results_cache.items()
                    if isinstance(res, dict) and 'future_dates' in res and res['future_dates']
                ]

                # 使用封装的函数，不直接写逻辑
                if indicators_with_prediction:
                    logger.info(f"UI: 找到 {len(indicators_with_prediction)} 个指标的 MOGP 预测结果")
                    
                    # === 调用封装函数渲染核心指标卡片 ===
                    # 传递必要的数据
                    render_mogp_core_metrics(
                        indicators_with_prediction=indicators_with_prediction,
                        mogp_results_cache=mogp_results_cache,
                        patient_data_raw=patient_data_raw,
                        mogp_last_updated=mogp_last_updated
                    )
                    
                    # === 图表和指标选择器（保持原样）===
                    st.markdown("---")
                    
                    # 初始化 session_state（用于记住用户的选择）
                    if 'selected_mogp_indicators' not in st.session_state:
                        st.session_state.selected_mogp_indicators = indicators_with_prediction[:1]
                    
                    # 先渲染选中的图表
                    current_selected = st.session_state.get('selected_mogp_indicators', indicators_with_prediction[:1])
                    
                    if current_selected:
                        # 使用 Tabs 展示每个指标的预测图
                        indicator_tabs = st.tabs([f"📊 {ind}" for ind in current_selected])
                        
                        for idx, indicator in enumerate(current_selected):
                            with indicator_tabs[idx]:
                                res = mogp_results_cache[indicator]
                                try:
                                    # 准备绘图数据（只绘制稳定期数据+预测）
                                    surveillance_data = patient_data[patient_data['phase'] == '稳定监控期']
                                    
                                    if not surveillance_data.empty and indicator in surveillance_data.columns:
                                        # 获取全局最后日期
                                        if not patient_data.empty:
                                            overall_last_date = patient_data.index[-1]
                                        else:
                                            overall_last_date = None
                                        
                                        forecast_fig = create_short_term_forecast_chart(
                                            historical_data=surveillance_data[[indicator]].dropna(),
                                            indicator=indicator,
                                            mogp_result=res,
                                            ref_range=ref_ranges_dict.get(indicator, (None, None)),
                                            overall_last_date=overall_last_date  # 传递全局最后日期
                                        )
                                        st.plotly_chart(forecast_fig, use_container_width=True, config={'displayModeBar': False})
                                        
                                        # 显示该指标的特定警告
                                        ind_warning = res.get('warning')
                                        if ind_warning:
                                            ind_warning_cn = {
                                                '⚠️ 不确定性快速增长，预测仅供参考': '⚠️ 数据波动较大，预测仅供参考',
                                                'ℹ️ 历史拟合误差相对较大，请结合基线观察': 'ℹ️ 数据拟合度一般，请结合基线观察'
                                            }.get(ind_warning, ind_warning)
                                            st.caption(f"💡 {ind_warning_cn}")
                                    else:
                                        if surveillance_data.empty:
                                            st.warning(
                                                f"⚠️ 指标 '{indicator}' 无稳定监控期数据。请确保：\n"
                                                "1. 至少有3次稳定期检测记录\n"
                                                "2. 数据中正确标记了'phase'列"
                                            )
                                        else:
                                            st.warning(f"⚠️ 指标 '{indicator}' 在稳定期数据中不存在（可能全为空值）")
                                
                                except Exception as plot_e:
                                    logger.error(f"绘制 MOGP 预测图失败 ({indicator}): {plot_e}", exc_info=True)
                                    st.error(f"图表渲染失败: {str(plot_e)}")
                    else:
                        st.info(language_support.t("choose_at_least_one_indicator"))
                    
                    # === 指标选择器放在图表下方 ===
                    st.markdown("---")
                    st.markdown(language_support.t("select_indicator"))
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        selected_indicators = st.multiselect(
                            language_support.t("multi_select_hint"),
                            options=indicators_with_prediction,
                            default=current_selected,
                            max_selections=4,
                            key="mogp_indicator_selector",
                            label_visibility="collapsed",
                            on_change=lambda: st.session_state.update({
                                'selected_mogp_indicators': st.session_state.mogp_indicator_selector
                            })
                        )
                    
                    with col2:
                        remaining_count = len(indicators_with_prediction) - len(selected_indicators)
                        if remaining_count > 0:
                            st.caption(language_support.t("remaining_indicators", count=remaining_count))
                        else:
                            st.caption(language_support.t("all_selected"))
                    
                    # 显示提示
                    if selected_indicators and len(selected_indicators) < len(indicators_with_prediction):
                        remaining = [ind for ind in indicators_with_prediction if ind not in selected_indicators]
                        st.caption(f"💡 还可选择: {', '.join(remaining[:3])}{'等' if len(remaining) > 3 else ''}")
                    
                else:
                    logger.info("UI: 未找到有效的 MOGP 预测结果。")
                    # 显示 MOGP 状态信息（即使没有预测结果）
                    render_mogp_status_card(mogp_diagnostic, False)
                    st.info(language_support.t("no_prediction"))

                st.markdown("---")

                # --- 第3层：其他观察（治疗反应、历史模式等）---
                other_observations = [
                    obs for obs in observation_result.get('observations', [])
                    if obs.get('pattern_type') != 'baseline_deviation' # 排除已显示的基线警报
                ]

                # 只计算传统观察项，SHAP 稍后动态判断
                total_observations = len(other_observations)

                # 检查是否有 SHAP 分析的必要条件

                has_shap_analysis = False
                has_correlation_analysis = False  # 单独判断相关性分析

                if not patient_data.empty:
                    latest_data_point_raw = patient_data.iloc[-1:]
                    latest_phase = latest_data_point_raw['phase'].iloc[0]
                    
                    models = health_controller.models or {}
                    model_pack = models.get(latest_phase)
                    
                    # 完整性检查
                    if (model_pack and isinstance(model_pack, dict) 
                        and 'model' in model_pack 
                        and model_pack.get('background_data') is not None
                        and model_pack.get('features_trained')):
                        has_shap_analysis = True
                        total_observations += 1  # 将 SHAP 分析算作一个"观察项"
                        logger.debug("✅ 检测到可用模型，将渲染 SHAP 分析")
                    else:
                        logger.debug("⚠️ 模型数据不完整或不存在，跳过 SHAP 分析")

                    # 独立判断相关性分析的可行性
                    original_lab_indicators = _get_original_lab_indicators()
                    if len(patient_data) > 5 and original_lab_indicators:
                        has_correlation_analysis = True
                        total_observations += 1  # 相关性分析算一项
                        logger.debug("✅ 历史数据充足，将渲染相关性分析")
                    else:
                        logger.debug("⚠️ 历史数据不足或无原始指标，跳过相关性分析")


                # === 渲染折叠项 ===
                if other_observations or has_shap_analysis or has_correlation_analysis:
                    logger.debug(f"UI: 渲染 {len(other_observations)} 个传统观察项 + {'SHAP分析' if has_shap_analysis else '无SHAP'} + {'相关性分析' if has_correlation_analysis else '无相关性分析'}")

                    with st.expander(f"📋 其他观察 ({total_observations} 项)", expanded=False):
                        # --- Part 1: 传统观察项 ---
                        if other_observations:
                            for obs in other_observations:
                                level = obs.get('attention_level', 'info')
                                level_emoji = {
                                    'high': '🔴', 
                                    'medium': '🟡', 
                                    'low': '🟢', 
                                    'info': 'ℹ️'
                                }
                                emoji = level_emoji.get(level, '⚪')

                                st.markdown(
                                    f"{emoji} **{obs.get('indicator', '综合')}**: {obs.get('observation', '')}"
                                )

                                # 显示 Data Context 和 Note (如果存在)
                                if obs.get('data_context'):
                                    st.caption(f"   数据: {obs['data_context']}")
                                # 统一处理 Note 和反馈引导
                                if obs.get('note'):
                                    # 检查是否是相似度相关的观察项（通过 pattern_type 判断）
                                    is_similarity_obs = (
                                        'unified_feedback_request' in obs or 
                                        obs.get('pattern_type') in ['similar_pattern', 'historical_match']
                                    )
                                    
                                    if is_similarity_obs:
                                        # 相似度观察：引导到历史回顾模块
                                        st.caption("   ↓ 请在下方「💭 帮助 AI 理解您的历史模式」中标记此日期")
                                    else:
                                        # 其他观察：保持原样显示 note
                                        st.caption(f"   💡 {obs['note']}")

                                #  处理内嵌反馈按钮（AI 发现的异常）
                                if obs.get('attention_level') in ['high', 'medium'] and obs.get('report_uuid'):
                                    # 获取必要的上下文数据
                                    all_labels = observation_result.get('context', {}).get('all_labels', {})
                                    obs_uuid = obs['report_uuid']
                                    if obs_uuid not in all_labels:
                                        st.caption("   ↓ 请在下方「💭 帮助 AI 理解您的历史模式」中标记此日期")
           
                        # --- Part 2: SHAP 分析 ---
                        if has_shap_analysis:
                            st.markdown("---")
                            st.markdown("##### 🔬 化验指标相互作用分析")
                            st.caption("显示您的基础化验指标（如血细胞、生化指标等）如何相互影响健康评分")
                            
                            try:
                                # 1. 获取必要数据
                                model = model_pack['model']
                                background_data_weighted = model_pack.get('background_data')
                                imputation_values = model_pack.get('imputation', pd.Series(dtype=float))
                                features_trained = model_pack.get('features_trained', [])
                                
                                if not features_trained or background_data_weighted is None or background_data_weighted.empty:
                                    st.info("💡 模型数据不完整（缺少特征列表或背景数据），无法进行分析。")
                                else:
                                    logger.info(f"SHAP 原始指标分析：模型已加载 (N={len(features_trained)} features)")
                                
                                    # 2. 获取权重
                                    all_weights = health_controller.weights
                                    if all_weights is None:
                                        st.warning("⚠️ 权重数据缺失，无法进行分析")
                                    else:
                                        # 3. 准备和验证数据
                                        data_point_aligned = latest_data_point_raw.reindex(
                                            columns=features_trained
                                        ).fillna(imputation_values).fillna(0)
                                        
                                        aligned_data, aligned_weights = data_point_aligned.align(
                                            all_weights, axis=1, join='inner'
                                        )
                                        
                                        weighted_data_point = aligned_data * aligned_weights
                                        
                                        if weighted_data_point.shape[1] != background_data_weighted.shape[1]:
                                            logger.error(
                                                                    f"❌ SHAP 特征提示：特征数量不匹配！"
                                                f"Model/Background data has {background_data_weighted.shape[1]} features, "
                                                f"but input data point has {weighted_data_point.shape[1]} features."
                                            )
                                            st.error("SHAP 分析失败：模型与数据特征不匹配。")
                                        else:
                                            logger.debug("🔍 SHAP 数据验证 (完整 N 特征):")
                                            logger.debug(f"  - background_data_weighted: {background_data_weighted.shape}")
                                            logger.debug(f"  - weighted_data_point: {weighted_data_point.shape}")

                                            # 4. 计算 SHAP
                                            with st.spinner("⏳ 正在分析指标..."):
                                                shap_results = explainability.get_shap_explanation(
                                                    model, 
                                                    background_data_weighted,
                                                    weighted_data_point,
                                                    compute_interactions=True
                                                )
                                            
                                            # 5. 筛选和绘图
                                            if shap_results and "shap_values_obj" in shap_results:
                                                full_shap_values = shap_results["shap_values_obj"]
                                                original_lab_indicators = _get_original_lab_indicators()
                                                
                                                features_to_plot_names = [
                                                    f for f in full_shap_values.feature_names 
                                                    if f in original_lab_indicators
                                                ]
                                                
                                                if not features_to_plot_names:
                                                    st.info("💡 当前分析中未涉及原始化验指标。")
                                                else:
                                                    shap_values_subset = full_shap_values[0, features_to_plot_names]
                                                    
                                                    logger.info(f"SHAP 结果筛选：从 {len(full_shap_values.feature_names)} 个总特征中，筛选出 {len(features_to_plot_names)} 个原始指标用于绘图。")

                                                    # === 交互图 ===
                                                    st.markdown("---")
                                                    st.markdown("#### 📊 指标相关性分析")
                                                    st.caption("💡 基于您的历史数据，以下指标对显示出统计关联性")                                     
                                                    shap_interaction_values = shap_results.get('shap_interaction_values')
                                                    
                                                    # 核心逻辑：无论交互值是否有效，都展示替代分析
                                                    if shap_interaction_values is not None:
                                                        # 尝试查找交互（虽然 IsolationForest 通常为 0）
                                                        try:
                                                            feature1, feature2, interaction_value = explainability.find_top_interaction_features(
                                                                shap_interaction_values,
                                                                weighted_data_point.columns.tolist()
                                                            )
                                                            
                                                            # 检查是否找到有意义的交互
                                                            if feature1 != "N/A" and feature2 != "N/A" and abs(interaction_value) > 1e-6:
                                                                # 【罕见情况】找到了显著交互（保留原有逻辑）
                                                                st.success(f"""
                                                                **检测到显著的指标间协同效应**
                                                                
                                                                - **指标组合**：`{feature1}` ⚡ `{feature2}`
                                                                - **交互强度**：{abs(interaction_value):.4f}
                                                                """)
                                                                
                                                                dependence_fig = explainability.plot_dependence(
                                                                    shap_interaction_values,
                                                                    weighted_data_point,
                                                                    feature1,
                                                                    feature2
                                                                )
                                                                st.pyplot(dependence_fig, use_container_width=True)
                                                                plt.close(dependence_fig)
                                                                logger.info("✅ SHAP 依赖图渲染成功")
                                                                
                                                                with st.expander("💡 如何看懂这张图？", expanded=False):
                                                                    st.markdown(f"""
                                                                    **这张图显示了 {feature1} 与 {feature2} 之间的交互效应：**
                                                                    
                                                                    1. **横轴（X轴）**: {feature1} 的数值范围
                                                                    2. **纵轴（Y轴）**: 两者共同作用的风险贡献值
                                                                    3. **颜色**: {feature2} 的实际数值（红=高，蓝=低）
                                                                    """)
                                                        
                                                        except Exception as interact_e:
                                                            logger.warning(f"⚠️ 交互分析失败: {interact_e}", exc_info=True)
                                                    # 主流程：展示替代分析（无论交互值如何）
                                                    # 相关性分析
                                                    try:
                                                        # 获取原始化验指标列表
                                                        original_lab_indicators = _get_original_lab_indicators()
                                                        
                                                        if not patient_data.empty and len(patient_data) > 5:
                                                            # 调用相关性分析函数
                                                            corr_results = explainability.analyze_feature_correlations(
                                                                patient_data,
                                                                original_lab_indicators,
                                                                top_n=5
                                                            )
                                                            
                                                            if not corr_results.empty:
                                                                # 展示相关性表格
                                                                for idx, row in corr_results.iterrows():
                                                                    col1, col2, col3 = st.columns([3, 1, 1])
                                                                    
                                                                    with col1:
                                                                        st.write(f"{row['emoji']} **{row['feature1']}** ↔ **{row['feature2']}**")
                                                                    
                                                                    with col2:
                                                                        st.metric(
                                                                            label=row['direction'],
                                                                            value=f"{row['abs_correlation']:.3f}"
                                                                        )
                                                                    
                                                                    with col3:
                                                                        st.caption(f"{row['strength']}相关")
                                                                
                                                                # 说明文字
                                                                with st.expander("💡 如何理解相关性？", expanded=False):
                                                                    st.markdown("""
                                                                    **相关性含义**：
                                                                    - **强相关（>0.7）**：两个指标通常同时升高或降低，可能存在生理学关联
                                                                    - **中等相关（0.4-0.7）**：存在一定关联，但不完全同步
                                                                    - **弱相关（<0.4）**：关联较弱，可能是偶然波动
                                                                    
                                                                    **重要提示**：
                                                                    - 相关性 ≠ 因果关系
                                                                    - 本分析仅供参考，不能替代医学诊断
                                                                    - 如有疑问，请咨询医生
                                                                    """)
                                                            else:
                                                                st.info("未找到显著相关的指标对")
                                                        else:
                                                            st.info("历史数据不足，无法进行相关性分析（需要 >5 个数据点）")
                                                            
                                                    except Exception as corr_e:
                                                        logger.error(f"相关性分析失败: {corr_e}", exc_info=True)
                                                        st.warning("相关性分析暂时不可用，请稍后重试")

                                                    # === 贡献图 ===
                                                    st.markdown("##### 📊 各化验指标的影响力")
                                                    st.caption("显示每个基础指标如何影响健康评分（红色=升高风险，蓝色=降低风险）")

                                                    try:
                                                        waterfall_fig = explainability.plot_waterfall(shap_values_subset, max_display=20)
                                                        st.pyplot(waterfall_fig, use_container_width=True)
                                                        plt.close(waterfall_fig)
                                                        logger.info("✅ SHAP 瀑布图（原始指标）渲染成功")
                                                        
                                                        # === 可选：提供详细数据查看 ===
                                                        with st.expander("📋 查看详细数值", expanded=False):
                                                            shap_values_obj = shap_results.get('shap_values_obj')
                                                            
                                                            if shap_values_obj is not None:
                                                                top_features = explainability.get_top_shap_features(
                                                                    shap_values_obj,
                                                                    original_lab_indicators,
                                                                    top_n=20  # 与图表数量一致
                                                                )
                                                                
                                                                if not top_features.empty:
                                                                    # 美化表格显示
                                                                    display_df = top_features.copy()
                                                                    display_df['排名'] = range(1, len(display_df) + 1)
                                                                    display_df = display_df[['排名', 'feature', 'impact']]
                                                                    display_df.columns = ['排名', '指标', '影响力']
                                                                    
                                                                    st.dataframe(
                                                                        display_df,
                                                                        hide_index=True,
                                                                        column_config={
                                                                            "排名": st.column_config.NumberColumn(format="%d"),
                                                                            "影响力": st.column_config.NumberColumn(format="%.4f")
                                                                        },
                                                                        use_container_width=True
                                                                    )
                                                                    

                                                    except Exception as e:
                                                        logger.warning(f"⚠️ 瀑布图渲染失败: {e}", exc_info=True)
                                                        st.warning(f"瀑布图渲染失败: {e}")
                                                
                                                                           
                            except Exception as e:
                                logger.error(f"❌ SHAP 原始指标分析失败: {e}", exc_info=True)
                                st.error(f"分析时出错: {str(e)}")
                                st.code(traceback.format_exc())
                        else:
                            # SHAP 不可用时的提示（可选）
                            if not other_observations:
                                st.info("💡 暂无观察项和可用模型")

                else:
                    # 既没有传统观察项，也没有 SHAP
                    logger.debug("UI: 没有观察项需要渲染。")

                # --- 第4层：统一反馈模块（V7.1 合并版）---
                logger.debug("UI: 调用 render_unified_feedback_module (V7.1 合并版)...")
                render_unified_feedback_module(observation_result, patient_id)

                # --- 底部：操作按钮 ---
                st.markdown("---")
                st.subheader(language_support.t("system_actions"))
                col1, col2 = st.columns(2)

                with col1:
                    if st.button(language_support.t("retrain_model"), use_container_width=True, help=language_support.t("retrain_help")):
                        logger.info("用户点击 '重新训练模型'")
                        with st.spinner(language_support.t("training_model")):
                            try:
                                train_result = app_core.train_models(health_controller) # train_models 内部会保存MOGP结果

                                if train_result.get('success'):
                                    logger.info("模型训练成功，准备更新UI状态...")

                                    # 检查是否使用了回退模式
                                    if train_result.get('fallback_activated'):
                                        logger.warning("训练使用了回退模式 (默认权重)")
                                        st.warning(
                                            "⚠️ **当前使用默认权重**\n\n"
                                            "系统检测到您的数据中缺少'强效治疗期'或'稳定监控期'的记录,无法计算个性化权重。\n\n"
                                            "**提示**:\n"
                                            "1. 补充至少5次'强效治疗期'的检测数据\n"
                                            "2. 补充至少5次'稳定监控期'的检测数据\n"
                                            "3. 然后重新点击'重新训练模型'\n\n"
                                            "这样AI才能学习您的个人健康模式。"
                                        )

                                    # 更新 session state 中的 MOGP 缓存
                                    mogp_sub_result = train_result.get('mogp', {})
                                    if mogp_sub_result.get('success'):
                                        st.session_state.mogp_results = mogp_sub_result.get('results')
                                        st.session_state.mogp_last_updated = datetime.now()
                                        st.session_state.mogp_target_indicators = mogp_sub_result.get('indicators')
                                        st.session_state.mogp_diagnostic_info = mogp_sub_result.get('diagnostic_info')
                                        logger.info("MOGP 结果已更新到 session_state")
                                    else:
                                        # 如果 MOGP 失败或未运行，清除缓存
                                        st.session_state.mogp_results = None
                                        st.session_state.mogp_last_updated = None
                                        logger.info("MOGP 未成功运行，相关缓存已清除")
                                    if train_result.get('fallback_activated'):
                                        st.info(language_support.t("training_done_fallback"))
                                    else:
                                        st.success(language_support.t("training_done"))
                                    # 短暂暂停后自动刷新页面以显示最新结果
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    logger.error(f"模型训练失败: {train_result.get('message', '未知错误')}")
                                    st.error(f"❌ 模型训练失败: {train_result.get('message', '请检查日志')}")

                            except Exception as train_e:
                                logger.error(f"调用 train_models 时发生严重错误: {train_e}", exc_info=True)
                                st.error(f"训练过程中发生错误: {train_e}")

                with col2:
                    if st.button(language_support.t("clear_cache"), use_container_width=True, help=language_support.t("clear_cache_help")):
                        logger.info("用户点击 '清除内部缓存'")
                        try:
                            health_controller.clear_cache()
                            # 同时清除 session state 中的 MOGP 缓存
                            st.session_state.mogp_results = None
                            st.session_state.mogp_last_updated = None
                            st.session_state.mogp_target_indicators = None
                            st.session_state.mogp_diagnostic_info = None
                            logger.info("Controller 缓存和 Session State MOGP 缓存已清除")
                            st.success(language_support.t("cache_cleared"))
                            time.sleep(2)
                            st.rerun()
                        except Exception as clear_e:
                            logger.error(f"清除缓存失败: {clear_e}", exc_info=True)
                            st.error(language_support.t("cache_clear_error"))

                # 保留导出按钮，但功能未实现
                # with col3:
                #     if st.button("📥 导出报告", use_container_width=True):
                #         st.info("📧 报告导出功能开发中...")

    # --- Tab 2: 录入数据 ---
    with data_entry:
        logger.debug("渲染 '录入数据' 选项卡。")
        st.header(language_support.t("data_entry_header"))
        
        if 'staged_items' not in st.session_state:
            st.session_state.staged_items = {}
        
        def auto_stage_current_data(items_df: pd.DataFrame):
            # 这是一个回调函数，日志记录器可能需要重新获取或传入
            logger_cb = logging.getLogger(__name__)

            if items_df is not None and not items_df.empty:
                valid_entries = items_df[pd.to_numeric(items_df["数值"], errors='coerce').notna()]
                for _, row in valid_entries.iterrows():
                    st.session_state.staged_items[row["指标"]] = {
                        "指标名称": row["指标"],
                        "检测值": float(row["数值"]),
                        "参考范围": row["正常范围"],
                        "单位": row["单位"]
                    }

        def save_or_queue_lab_report(final_df: pd.DataFrame, target_date_str: str, target_phase: str) -> bool:
            """保存一份报告；如果同日同指标冲突，则进入确认流程。"""
            patient_data_existing = data_manager.load_patient_data(patient_id)
            items_to_overwrite = []
            existing_rows = None

            if not patient_data_existing.empty:
                existing_dates = patient_data_existing.index.strftime("%Y-%m-%d").tolist()
                if target_date_str in existing_dates:
                    existing_rows = patient_data_existing[
                        patient_data_existing.index.strftime("%Y-%m-%d") == target_date_str
                    ]
                    for item_name in final_df["指标名称"].tolist():
                        if item_name in existing_rows.columns and existing_rows[item_name].notna().any():
                            items_to_overwrite.append(item_name)

            if items_to_overwrite:
                st.session_state.pending_save = {
                    'final_df': final_df,
                    'target_date_str': target_date_str,
                    'phase': target_phase,
                    'items_to_overwrite': items_to_overwrite,
                    'existing_rows': existing_rows
                }
                st.rerun()
                return False

            data_manager.save_or_merge_lab_report(patient_id, target_date_str, target_phase, final_df)
            st.session_state.data_changed = True
            health_controller.clear_cache()
            return True

        if 'ocr_report_drafts' not in st.session_state:
            st.session_state.ocr_report_drafts = []

        with st.expander(language_support.t("ocr_expander"), expanded=False):
            if ocr_importer.rapidocr_available():
                st.caption(language_support.t("ocr_privacy"))
                uploaded_report_images = st.file_uploader(
                    language_support.t("upload_report_images"),
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key="ocr_report_uploader"
                )

                if st.button(language_support.t("ocr_button"), disabled=not uploaded_report_images, key="ocr_parse_button"):
                    parsed_reports = []
                    with st.spinner(language_support.t("ocr_running")):
                        for uploaded_file in uploaded_report_images or []:
                            try:
                                ocr_lines = ocr_importer.run_ocr_on_image_bytes(uploaded_file.getvalue())
                                parsed = ocr_importer.parse_report_text_lines(
                                    ocr_lines,
                                    source_name=uploaded_file.name
                                )
                                parsed_reports.append(parsed)
                            except Exception as exc:
                                logger.error(f"OCR 识别失败: {uploaded_file.name}: {exc}", exc_info=True)
                                st.error(f"{uploaded_file.name} 识别失败：{exc}")

                    if parsed_reports:
                        st.session_state.ocr_report_drafts = parsed_reports
                        st.success(f"已识别 {len(parsed_reports)} 张图片，请逐项核对后保存。")

                for idx, draft in enumerate(st.session_state.ocr_report_drafts):
                    source_name = draft.get("source_name") or f"报告 {idx + 1}"
                    items_df = draft.get("items", pd.DataFrame())
                    with st.container(border=True):
                        st.markdown(f"#### {source_name}")
                        parsed_date = draft.get("report_date")
                        date_source = draft.get("date_source") or "未识别"
                        if parsed_date:
                            default_date = datetime.strptime(parsed_date, "%Y-%m-%d")
                            st.info(f"识别到检查日期：{parsed_date}（来源：{date_source}）。如上传了更早月份的报告，可直接保留该日期保存。")
                        else:
                            default_date = datetime.now()
                            st.warning("未识别到采样/报告日期，请手动选择检查日期后再保存。")

                        ocr_date = st.date_input(
                            language_support.t("ocr_date_label"),
                            value=default_date,
                            key=f"ocr_report_date_{idx}"
                        )
                        ocr_phase = st.radio(
                            language_support.t("health_phase"),
                            options=["稳定监控期", "强效治疗期"],
                            format_func=language_support.phase_label,
                            horizontal=True,
                            index=0,
                            key=f"ocr_phase_{idx}",
                        )

                        if items_df is None or items_df.empty:
                            st.warning("未匹配到可填充指标。可展开原始文本查看 OCR 是否识别成功。")
                        else:
                            editor_df = items_df.copy()
                            editor_df.insert(0, "保留", True)
                            edited_ocr_df = st.data_editor(
                                editor_df,
                                column_config={
                                    "保留": st.column_config.CheckboxColumn("保留", default=True),
                                    "指标名称": st.column_config.TextColumn("指标", disabled=True),
                                    "检测值": st.column_config.NumberColumn("检测值", required=True),
                                    "OCR文本": st.column_config.TextColumn("OCR 原文", disabled=True),
                                    "识别置信度": st.column_config.NumberColumn("置信度", disabled=True),
                                    "匹配方式": st.column_config.TextColumn("匹配", disabled=True),
                                },
                                hide_index=True,
                                width='stretch',
                                key=f"ocr_editor_{idx}"
                            )

                            col_save_ocr, col_stage_ocr = st.columns(2)
                            with col_save_ocr:
                                if st.button("💾 保存这份识别结果", type="primary", width='stretch', key=f"ocr_save_{idx}"):
                                    selected_rows = edited_ocr_df[
                                        (edited_ocr_df["保留"] == True) &
                                        (pd.to_numeric(edited_ocr_df["检测值"], errors="coerce").notna())
                                    ]
                                    final_df = selected_rows[["指标名称", "检测值"]].copy()
                                    if final_df.empty:
                                        st.warning("没有可保存的有效指标。")
                                    else:
                                        try:
                                            saved = save_or_queue_lab_report(
                                                final_df,
                                                ocr_date.strftime("%Y-%m-%d"),
                                                ocr_phase
                                            )
                                            if saved:
                                                st.success(f"已保存 {len(final_df)} 个指标。")
                                                time.sleep(1)
                                                st.rerun()
                                        except Exception as exc:
                                            logger.error(f"OCR 保存失败: {exc}", exc_info=True)
                                            st.error(f"保存失败：{exc}")

                            with col_stage_ocr:
                                if st.button("填入下方手动表单", width='stretch', key=f"ocr_stage_{idx}"):
                                    for _, row in edited_ocr_df[edited_ocr_df["保留"] == True].iterrows():
                                        if pd.notna(row["检测值"]):
                                            st.session_state.staged_items[row["指标名称"]] = {
                                                "指标名称": row["指标名称"],
                                                "检测值": float(row["检测值"]),
                                                "参考范围": "",
                                                "单位": ""
                                            }
                                    st.session_state.entry_report_date = ocr_date
                                    st.success("已填入下方手动表单，请继续核对后保存。")
                                    st.rerun()

                        unmatched_lines = draft.get("unmatched_lines", [])
                        with st.expander("查看 OCR 原始文本和未匹配行", expanded=False):
                            st.text(draft.get("raw_text", ""))
                            if unmatched_lines:
                                st.caption("未匹配行：")
                                for line in unmatched_lines[:30]:
                                    st.caption(f"- {line}")
            else:
                st.warning("当前环境未检测到 `opencv-python` 和 `rapidocr-onnxruntime`，无法启用本地图片识别。")
        
        if st.session_state.staged_items:
            st.success(language_support.t("filled_count", count=len(st.session_state.staged_items)))
            with st.expander(language_support.t("filled_preview"), expanded=False):
                preview_data = []
                for name, data in st.session_state.staged_items.items():
                    preview_data.append({
                        "指标": name,
                        "数值": data["检测值"],
                        "单位": data.get("单位", ""),
                    })
                preview_df = pd.DataFrame(preview_data).rename(columns={
                    "指标": language_support.t("indicator_column"),
                    "数值": language_support.t("raw_value_column"),
                    "单位": language_support.t("unit_column"),
                })
                st.dataframe(preview_df, hide_index=True, width='stretch')
        else:
            st.info(language_support.t("manual_entry_hint"))
        

        col1, col2 = st.columns(2)
        with col1:
            if 'entry_report_date' not in st.session_state:
                st.session_state.entry_report_date = datetime.now()
            report_date = st.date_input(language_support.t("report_date"), key="entry_report_date")
        with col2:
            st.markdown(language_support.t("health_phase"))
            # 使用 Radio Button，但变量名仍为 phase
            phase = st.radio(
                language_support.t("phase_context"),
                options=["稳定监控期", "强效治疗期"],
                format_func=language_support.phase_label,
                horizontal=True,
                index=0, # 默认选“稳定监控期”
                key="phase_tag_radio", # key 可以随意
                label_visibility="collapsed",
                help=language_support.t("phase_help")
            )

        st.markdown("---")
        
        st.markdown(language_support.t("fill_indicators"))
        
        if 'last_template' not in st.session_state:
            st.session_state.last_template = None
        
        selected_template = st.selectbox(
            language_support.t("select_report_type"),
            options=list(config.LAB_REPORT_CONFIG.keys()),
        )
        
        if st.session_state.last_template != selected_template:
            if 'current_editor_data' in st.session_state and st.session_state.current_editor_data is not None:
                logger.debug("自动暂存上一个模板的数据...")
                auto_stage_current_data(st.session_state.current_editor_data)
            st.session_state.last_template = selected_template
        
        items_for_editor = []
        if selected_template:
            template_items = config.LAB_REPORT_CONFIG[selected_template]
            for item in template_items:
                staged_value = st.session_state.staged_items.get(item["name"], {}).get("检测值", None)
                items_for_editor.append({
                    "指标": item["name"],
                    "数值": staged_value,
                    "正常范围": f"{item.get('lower', 'N/A')} - {item.get('upper', 'N/A')}",
                    "单位": item.get("unit", "")
                })
            
            edited_df = st.data_editor(
                pd.DataFrame(items_for_editor),
                column_config={
                    "指标": st.column_config.TextColumn(language_support.t("indicator_column"), disabled=True, width="medium"),
                    "数值": st.column_config.NumberColumn(
                        language_support.t("value_column"), 
                        required=False, 
                        width="small",
                        help=language_support.t("value_help")
                    ),
                    "正常范围": st.column_config.TextColumn(language_support.t("reference_range_column"), disabled=True, width="medium"),
                    "单位": st.column_config.TextColumn(language_support.t("unit_column"), disabled=True, width="small"),
                },
                hide_index=True,
                width='stretch',
                key=f"data_entry_editor_{selected_template}_{st.session_state.editor_reset_counter}"
            )
            
            st.session_state.current_editor_data = edited_df
        
        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            if st.button(language_support.t("save_report"), type="primary", width='stretch'):
                logger.info("用户点击 '保存化验单'。")
                # 1. 暂存当前编辑器数据
                if 'current_editor_data' in st.session_state and st.session_state.current_editor_data is not None:
                    logger.debug("暂存当前编辑器中的数据...")
                    auto_stage_current_data(st.session_state.current_editor_data)
                
                if st.session_state.staged_items:
                    # 2. 准备数据
                    final_data = []
                    for item_name, item_data in st.session_state.staged_items.items():
                        final_data.append({
                            "指标名称": item_data["指标名称"],
                            "检测值": item_data["检测值"]
                        })
                    final_df = pd.DataFrame(final_data)
                    target_date_str = report_date.strftime("%Y-%m-%d")

                    # 3. 检查冲突
                    patient_data_existing = data_manager.load_patient_data(patient_id)
                    items_to_overwrite = []
                    existing_rows = None

                    if not patient_data_existing.empty:
                        existing_dates = patient_data_existing.index.strftime("%Y-%m-%d").tolist()
                        if target_date_str in existing_dates:
                            # 获取该日期的所有记录（可能有多条）
                            existing_rows = patient_data_existing[
                                patient_data_existing.index.strftime("%Y-%m-%d") == target_date_str
                            ]
                            
                            # 健壮的冲突检测
                            for item_name in st.session_state.staged_items.keys():
                                # 检查该指标是否存在于历史数据的列中
                                if item_name in existing_rows.columns:
                                    # 只有当该列存在且有非空值时，才算冲突
                                    if (existing_rows[item_name].notna()).any():
                                        items_to_overwrite.append(item_name)
                                        logger.debug(f"  检测到冲突: {item_name}")
                                else:
                                    # 该指标是新指标，不算冲突
                                    logger.debug(f"  新增指标（无冲突）: {item_name}")

                    # 4. 决策：有冲突吗？
                    if items_to_overwrite:
                        st.session_state.pending_save = {
                            'final_df': final_df,
                            'target_date_str': target_date_str,
                            'phase': phase,
                            'items_to_overwrite': items_to_overwrite,
                            'existing_rows': existing_rows
                        }
                        st.rerun()
                    
                    else:
                        # 场景B：无冲突 → 直接保存
                        logger.info(f"无数据冲突，直接保存 {len(final_df)} 个指标。")
                        try:
                            data_manager.save_or_merge_lab_report(
                                patient_id, target_date_str, phase, final_df
                            )
                            st.success(language_support.t("save_success_count", count=len(final_df)))
                            
                            # 清理缓存
                            st.session_state.staged_items.clear()
                            st.session_state.current_editor_data = None
                            st.session_state.last_template = None
                            st.session_state.editor_reset_counter += 1 
                            st.session_state.data_changed = True
                            health_controller.clear_cache()
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            logger.error(f"保存化验单失败 (无冲突): {str(e)}", exc_info=True)
                            st.error(f"❌ 保存失败: {str(e)}")
                else:
                    logger.warning("用户点击保存，但 'staged_items' 为空。")
                    st.warning(language_support.t("need_one_indicator"))

        with col2:
            if st.button(language_support.t("clear_form"), width='stretch'):
                logger.info("用户点击 '清空重填'。")
                st.session_state.staged_items.clear()
                st.session_state.current_editor_data = None
                st.session_state.last_template = None
                st.session_state.editor_reset_counter += 1 
                st.rerun()

        with col3:
            st.caption(language_support.t("template_switch_hint"))

  
        if 'pending_save' in st.session_state and st.session_state.pending_save:
            logger.debug("渲染 'pending_save' (数据冲突) UI。")
            st.markdown("---")
            pending = st.session_state.pending_save
            
            st.warning(f"### ⚠️ 发现 {len(pending['items_to_overwrite'])} 项数据冲突")
            st.markdown(f"**{pending['target_date_str']}** 已存在以下指标，是否覆盖？")
            
            # 显示冲突对比
            comparison_data = []
            
            # 从保存快照中创建安全的新值查找字典
            new_values_map = {
                row['指标名称']: row['检测值'] 
                for _, row in pending['final_df'].iterrows()
            }
            
            # 日志记录（调试用）
            logger.debug(
                f"冲突检测：快照包含 {len(new_values_map)} 个指标，"
                f"待检查 {len(pending['items_to_overwrite'])} 个冲突项"
            )

            for indicator in pending['items_to_overwrite']:
                # 只处理快照中存在的指标
                if indicator not in new_values_map:
                    logger.warning(
                        f"⚠️ 冲突指标 '{indicator}' 未在保存快照中找到，"
                        f"可能是历史数据中的孤儿列（跳过显示）"
                    )
                    continue
                
                # 验证历史数据列存在性
                if indicator not in pending['existing_rows'].columns:
                    logger.warning(
                        f"⚠️ 冲突指标 '{indicator}' 不在历史数据列中，"
                        f"可能是数据结构变更（跳过显示）"
                    )
                    continue

                # 提取历史值
                existing_values = pending['existing_rows'][indicator].dropna()
                
                if existing_values.empty:
                    existing_values_str = "无有效数据"
                    logger.debug(f"  - {indicator}: 历史列存在但全为 NaN")
                else:
                    existing_values_str = ", ".join([f"{v:.2f}" for v in existing_values])
                
                # 从快照中安全获取新值
                new_value = new_values_map[indicator]
                new_value_str = f"{new_value:.2f}"

                comparison_data.append({
                    "指标": indicator,
                    "现有记录": f"{len(existing_values)} 条: {existing_values_str}",
                    "新值": new_value_str
                })
            
            # 渲染对比表或提示
            if comparison_data:
                st.dataframe(
                    pd.DataFrame(comparison_data), 
                    hide_index=True, 
                    width='stretch'
                )
                st.warning(
                    f"⚠️ 该日期 **{pending['target_date_str']}** 已有记录。\n\n"
                    f"继续保存将使用新数据 **覆盖或合并** 到该日期的同一份报告中。"
                )
            else:
                logger.warning(
                    f"❌ 对比表为空！items_to_overwrite={pending['items_to_overwrite']}, "
                    f"new_values_map keys={list(new_values_map.keys())}"
                )
                st.info(
                    "💡 系统检测到潜在冲突，但无法生成对比表。\n\n"
                    "**可能原因**：\n"
                    "- 历史数据列与当前数据不匹配\n"
                    "- 数据库中存在孤儿列\n\n"
                    "**提示**：点击「✅ 确认覆盖」强制合并，或「❌ 取消」后重新整理数据。"
                )
                        
            # 提示新增指标
            pending_item_names = set(pending['final_df']['指标名称'].tolist())
            items_to_add = [
                name for name in pending_item_names
                if name not in pending['items_to_overwrite']
            ]
            if items_to_add:
                st.info(f"✅ 将同时新增 {len(items_to_add)} 个新指标 (如: {', '.join(items_to_add[:3])}...)")
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("✅ 确认覆盖", type="primary", width='stretch', key="confirm_merge"):
                    try:
                        data_manager.save_or_merge_lab_report(
                            patient_id,
                            pending['target_date_str'],
                            pending['phase'],
                            pending['final_df']
                        )
                        st.success("✅ 保存成功!")
                        
                        # 清理所有缓存
                        st.session_state.staged_items.clear()
                        st.session_state.current_editor_data = None
                        st.session_state.last_template = None
                        st.session_state.data_changed = True
                        del st.session_state.pending_save
                        st.session_state.editor_reset_counter += 1
                        health_controller.clear_cache()
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        logger.error(f"保存化验单失败 (有冲突): {str(e)}", exc_info=True)
                        st.error(f"❌ 保存失败: {str(e)}")
            
            with col_b:
                if st.button("❌ 取消", width='stretch', key="cancel_merge"):
                    logger.info("用户点击 '取消' (合并/覆盖)。")
                    del st.session_state.pending_save
                    st.info("已取消")
                    st.rerun()

    # --- Tab 3: 历史记录 ---
    with data_management:
        st.header(language_support.t("history_header"))
        
        history_tab, settings_tab = st.tabs([
            language_support.t("history_records"),
            language_support.t("reference_ranges"),
        ])
        
        with history_tab:
            if 'editing_report_id' not in st.session_state:
                st.session_state.editing_report_id = None
            
            patient_data_raw_history, _, _ = health_controller.get_processed_data()
            
            if st.session_state.editing_report_id is None:

                if not patient_data_raw_history.empty:
                    st.info(language_support.t("expand_record_hint"))
                    for _, row_data in patient_data_raw_history.iterrows():

                        # 显示 phase 和 user_label
                        LABEL_MAP_CHINESE = {
                            'benign': '良性波动',
                            'significant': '重要变化',
                            'lab_error': '数据错误'
                        }
                        
                        # 从数据库获取内部标签 (e.g., 'benign')
                        label_str_internal = row_data.get('user_label') 
                        
                        # 查找对应的中文显示 (e.g., '良性波动')
                        label_str_display = LABEL_MAP_CHINESE.get(label_str_internal)
                        
                        # 仅当标签存在且映射成功时，才显示中文标签
                        label_display = f" | 🏷️ {label_str_display}" if label_str_display else ""

                        phase_display = language_support.phase_label(row_data['phase'])
                        with st.expander(f"📅 {row_data.name.strftime('%Y年%m月%d日')} - {phase_display}{label_display}"):
                            display_data = row_data.drop(['report_uuid', 'phase', 'user_label'], errors='ignore').dropna().to_frame('数值')
                            st.dataframe(display_data, width='stretch')
                            
                            col1, col2 = st.columns(2)
                            if col1.button(language_support.t("edit"), key=f"edit_{row_data['report_uuid']}", width='stretch'):
                                logger.info(f"用户点击 '修改' 按钮, report_uuid: {row_data['report_uuid']}")
                                st.session_state.editing_report_id = row_data['report_uuid']
                                # 检查是否有标签
                                current_label = row_data.get('user_label')
                                st.session_state.editing_report_has_label = pd.notna(current_label)
                                st.session_state.editing_date = row_data.name.date()
                                st.session_state.editing_phase = row_data['phase']
                                
                                items_for_editor = []
                                numeric_data = row_data.drop(['report_uuid', 'phase', 'user_label'], errors='ignore')

                                for item_name, item_value in numeric_data.items():
                                    if pd.notna(item_value):  # 只添加有值的指标
                                        items_for_editor.append({
                                            "指标名称": item_name, 
                                            "检测值": float(item_value)
                                        })
                                
                                st.session_state.editing_data = pd.DataFrame(items_for_editor)
                                health_controller.clear_cache()
                                st.rerun()

                            if col2.button(language_support.t("delete"), key=f"del_{row_data['report_uuid']}", width='stretch'):
                                logger.info(f"用户点击 '删除' 按钮, report_uuid: {row_data['report_uuid']}")
                                data_manager.delete_lab_report(row_data['report_uuid'])
                                st.session_state.data_changed = True
                                health_controller.clear_cache()
                                st.success(language_support.t("deleted"))
                                logger.info(f"报告 {row_data['report_uuid']} 已删除。")
                                st.rerun()
                else:
                    st.info(language_support.t("no_records"))

            else: # 编辑模式
                editing_uuid = st.session_state.editing_report_id
                logger.debug(f"处于 '编辑记录' 模式, report_uuid: {editing_uuid}")
                st.subheader(f"修改 {st.session_state.editing_date.strftime('%Y年%m月%d日')} 的记录")

                if st.session_state.get('editing_report_has_label'):
                    st.warning(
                        "⚠️ **注意**：此报告已有反馈标签。保存修改后，标签将被清除"
                        "（因为数据已变更）。如需保留标签，请取消修改。"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    new_report_date = st.date_input(
                        "检查日期", 
                        value=st.session_state.editing_date,
                        key="edit_date_picker"
                    )
                with col2:
                    phase_idx = 0 if st.session_state.editing_phase == "稳定监控期" else 1
                    new_phase = st.selectbox(
                        language_support.t("phase_context"), 
                        ["稳定监控期", "强效治疗期"], 
                        index=phase_idx,
                        format_func=language_support.phase_label,
                        key="edit_phase_selector"
                    )
                
                if 'editing_data' not in st.session_state or st.session_state.editing_data.empty:
                    logger.warning("进入编辑模式，但 'editing_data' 为空。")
                    st.warning("编辑数据丢失,请返回重试")
                    if st.button("❌ 返回", width='stretch'):
                        st.session_state.editing_report_id = None
                        st.rerun()
                else:
                    edited_items_df = st.data_editor(
                        st.session_state.editing_data,
                        column_config={
                            "指标名称": st.column_config.TextColumn(disabled=True, width="medium"),
                            "检测值": st.column_config.NumberColumn(
                                "检测值", 
                                required=True, 
                                width="small",
                                format="%.2f"
                            ),
                        },
                        hide_index=True,
                        width='stretch',
                        num_rows="dynamic",
                        key="edit_data_editor"
                    )
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 保存修改", width='stretch', type="primary"):
                            logger.info(f"用户点击 '保存修改', report_uuid: {editing_uuid}")
                            valid_items = edited_items_df[
                                edited_items_df["指标名称"].str.strip().astype(bool) & 
                                pd.to_numeric(edited_items_df["检测值"], errors='coerce').notna()
                            ].copy()
                            
                            if valid_items.empty:
                                st.error("❌ 没有有效数据可保存")
                            else:
                                try:
                                    valid_items["检测值"] = pd.to_numeric(valid_items["检测值"])
                                    
                                    data_manager.update_lab_report(
                                        patient_id,
                                        editing_uuid, 
                                        new_report_date.strftime("%Y-%m-%d"),
                                        new_phase,
                                        valid_items
                                    )

                                    logger.info(f"报告 {editing_uuid} 更新成功。")
                                    st.success("✅ 保存成功!")
                                    st.session_state.data_changed = True
                                    st.session_state.editing_report_id = None
                                    health_controller.clear_cache()
                                    if 'editing_data' in st.session_state:
                                        del st.session_state.editing_data
                                    if 'editing_date' in st.session_state:
                                        del st.session_state.editing_date
                                    if 'editing_phase' in st.session_state:
                                        del st.session_state.editing_phase
                                    
                                    time.sleep(2)
                                    st.rerun()
                                    
                                except Exception as e:
                                    logger.error(f"更新报告 {editing_uuid} 失败: {str(e)}", exc_info=True)
                                    st.error(f"❌ 保存失败: {str(e)}")
                                    st.code(traceback.format_exc())
                    
                    with col2:
                        if st.button("❌ 取消", width='stretch'):
                            st.session_state.editing_report_id = None
                            if 'editing_data' in st.session_state:
                                del st.session_state.editing_data
                            if 'editing_date' in st.session_state:
                                del st.session_state.editing_date
                            if 'editing_phase' in st.session_state:
                                del st.session_state.editing_phase
                            st.rerun()
        
        with settings_tab:
            logger.debug("渲染 '参考范围设置' 子选项卡。")
            st.subheader(language_support.t("reference_range_header"))
            
            if st.button(language_support.t("import_defaults")):
                logger.info("用户点击 '从系统导入默认值' (参考范围)。")

                default_refs_list = []
                for items in config.LAB_REPORT_CONFIG.values():
                    for item in items:
                        if item.get("lower") is not None and item.get("upper") is not None:
                            default_refs_list.append({
                                "指标名称": item["name"],
                                "lower_bound": item["lower"],
                                "upper_bound": item["upper"]
                            })
                
                if default_refs_list:
                    default_refs_df = pd.DataFrame(default_refs_list).drop_duplicates(
                        subset=["指标名称"]
                    ).set_index("指标名称")
                    data_manager.save_or_update_references(default_refs_df)
                    st.success(language_support.t("import_success"))
                    st.session_state.data_changed = True
                    health_controller.clear_cache()
                    st.rerun()
            
            st.markdown("---")
            
            saved_refs_df = data_manager.load_references()
            if not patient_data_raw.empty:
                historical_items = [
                    col for col in patient_data_raw.select_dtypes(include=np.number).columns 
                    if col not in ['report_uuid']
                ]
            else:
                historical_items = []
            
            config_refs_list = []
            for items in config.LAB_REPORT_CONFIG.values():
                for item in items:
                    config_refs_list.append({
                        "指标名称": item["name"],
                        "lower_bound": item.get("lower"),
                        "upper_bound": item.get("upper")
                    })
            config_refs_df = pd.DataFrame(config_refs_list).drop_duplicates(
                subset=["指标名称"]
            ).set_index("指标名称")
            
            all_known_items = sorted(list(set(
                historical_items + saved_refs_df.index.tolist() + config_refs_df.index.tolist()
            )))
            
            display_df = pd.DataFrame(index=all_known_items)
            display_df.index.name = "指标名称"
            display_df = display_df.join(config_refs_df)
            display_df.update(saved_refs_df)
            
            edited_refs_df = st.data_editor(
                display_df.reset_index(),
                column_config={
                    "指标名称": st.column_config.TextColumn(disabled=True),
                    "lower_bound": st.column_config.NumberColumn("下限", format="%.2f"),
                    "upper_bound": st.column_config.NumberColumn("上限", format="%.2f"),
                },
                hide_index=True,
                width='stretch',
                key="refs_editor"
            )
            
            if st.button(language_support.t("save_settings"), type="primary", width='stretch'):
                logger.info("用户点击 '保存设置' (参考范围)。")
                valid_refs = edited_refs_df.dropna(
                    subset=['lower_bound', 'upper_bound'], 
                    how='all'
                ).set_index('指标名称')
                
                if not valid_refs.empty:
                    logger.debug(f"正在保存 {len(valid_refs)} 条参考范围。")
                    data_manager.save_or_update_references(valid_refs)
                    st.success(language_support.t("save_success"))
                    st.session_state.data_changed = True
                    health_controller.clear_cache()
                    st.rerun()
                else:
                    logger.warning("用户尝试保存参考范围，但没有有效数据。")
                    st.warning(language_support.t("no_save_data"))


    # --- Tab 4: 适应性模拟 ---
    # --- Tab 4: 适应性模拟 ---
    with simulation_tab:
        st.header(language_support.t("simulation_header"))
        
        st.info(language_support.t("simulation_intro"))
        
        # === 前置检查 ===
        if patient_data_raw.empty:
            st.warning(language_support.t("add_first_record"))
        else:
            # ========================================
            # 第 1️⃣ 部分：选择机理模型
            # ========================================
            st.markdown("---")
            st.subheader(language_support.t("choose_model"))
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                selected_model = st.selectbox(
                    language_support.t("choose_model_select"),
                    options=list(config.MECHANISTIC_MODELS.keys()),
                    help=language_support.t("choose_model_help")
                )
            
            with col2:
                model_info = config.MECHANISTIC_MODELS[selected_model]
                st.info(f"""
                **{selected_model}**
                
                {model_info['description']}
                
                **核心特性**: {', '.join(model_info['key_features'])}
                
                **典型应用**: {model_info['typical_use']}
                """)

            # ========================================
            # 第 2️⃣ 部分：设置初始状态（✅ 提前到校准之前）
            # ========================================
            st.markdown("---")
            st.subheader(language_support.t("initial_state"))
            
            # 获取最新的肿瘤标志物值
            tumor_markers = [
                item['name'] for item in config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
            ]
            
            available_markers = [m for m in tumor_markers if m in patient_data_raw.columns]
            
            if not available_markers:
                st.error("❌ 未找到肿瘤标志物数据，请先录入 CEA、CA199 等指标")
                st.stop()  # ✅ 直接终止，避免后续代码执行
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_marker = st.selectbox(
                        "选择用于模拟的标志物",
                        options=available_markers,
                        key="sim_marker_selector"  # ✅ 添加唯一key
                    )
                
                with col2:
                    latest_value = patient_data_raw[selected_marker].dropna().iloc[-1]
                    st.metric("当前值", f"{latest_value:.2f}")
                
                with col3:
                    ref_upper = ref_ranges_dict.get(selected_marker, (None, None))[1]
                    if ref_upper:
                        st.metric("参考上限", f"{ref_upper:.2f}")
                    else:
                        st.caption("无参考上限")

                host_context_preview = simulation_engine.build_host_context(patient_data_raw)
                with st.expander("🧭 基础化验状态（用于模拟区间，不用于给药决定）", expanded=False):
                    host_indices = host_context_preview.get("indices", {})
                    host_scores = host_context_preview.get("scores", {})

                    score_cols = st.columns(4)
                    with score_cols[0]:
                        st.metric("炎症负荷代理", f"{host_scores.get('inflammation_load', 0):.2f}")
                    with score_cols[1]:
                        st.metric("免疫储备代理", f"{host_scores.get('immune_reserve', 0):.2f}")
                    with score_cols[2]:
                        st.metric("营养储备代理", f"{host_scores.get('nutrition_reserve', 0):.2f}")
                    with score_cols[3]:
                        st.metric("数据约束强度", f"{host_scores.get('data_confidence', 0):.2f}")

                    compact_indices = {
                        name: value for name, value in host_indices.items()
                        if value is not None and name in ["NLR", "PLR", "LMR", "SII", "SIRI", "PNI", "mGPS"]
                    }
                    if compact_indices:
                        st.json({k: round(v, 3) if isinstance(v, (int, float, np.floating)) else v for k, v in compact_indices.items()})

                    for obs in host_context_preview.get("observations", []):
                        st.caption(f"- {obs}")

                    st.caption("这些指标受感染、近期治疗、营养状态、采样时间等影响；这里只用于让模拟区间更诚实，而不是生成治疗指令。")

                # ========================================
                # 第 3️⃣ 部分：模型个性化校准（可选）
                # ========================================
                st.markdown("---")
                st.subheader(language_support.t("personal_calibration"))
                
                # 说明文字
                with st.expander("💡 为什么需要校准？", expanded=False):
                    st.markdown("""
                    **文献模型 vs 您的实际情况**
                    
                    机理模型的参数来自科学文献，基于"人群平均值"。但每个人的肿瘤特性不同：
                    - 生长速度可能更快/更慢
                    - 对药物的敏感性不同
                    - 免疫系统的状态不同
                    
                    **校准如何工作**
                    
                    1. 系统用您的前 N-1 次检测数据"训练"模型参数
                    2. 用最后一次检测数据"验证"预测精度
                    3. 如果误差 <30%，则认为校准成功
                    
                    **校准后的作用**
                    
                    - 预测更贴近您的实际情况（而非平均人群）
                    - 能让模拟更贴近您的数据背景
                    - 下次模拟会自动使用校准参数
                    
                    **数据要求**
                    
                    - 至少需要 3 次该标志物的检测记录
                    - 数据跨度最好 >3 个月
                    - 包含治疗期和稳定期的数据效果更好
                    """)
                
                # === 校准状态显示 ===
                col_status, col_action = st.columns([3, 2])
                
                with col_status:
                    # 检查是否已有校准参数
                    calibration_status = data_manager.load_calibrated_params(
                        patient_id, selected_model, selected_marker
                    )
                    
                    if calibration_status and calibration_status['is_reliable']:
                        st.success(
                            f"✅ **已使用个性化参数**\n\n"
                            f"- 验证误差: {calibration_status['error']:.1%}\n"
                            f"- 验证日期: {calibration_status['date']}\n"
                            f"- 预测值: {calibration_status.get('predicted_value', 0):.2f}\n"
                            f"- 实际值: {calibration_status.get('actual_value', 0):.2f}\n"
                            f"- 保存时间: {calibration_status.get('timestamp', 'N/A')[:10]}"
                        )
                    elif calibration_status:
                        st.warning(
                            f"⚠️ **校准参数存在但不可靠**\n\n"
                            f"- 验证误差: {calibration_status['error']:.1%}（>30%）\n"
                            f"- 当前使用: 文献默认参数\n\n"
                            f"提示：可重新校准或积累更多数据"
                        )
                    else:
                        st.info(
                            f"ℹ️ **使用文献默认参数**\n\n"
                            f"点击右侧按钮进行校准\n"
                            f"（需要至少3个数据点）"
                        )
                
                with col_action:
                    # 检查是否有足够数据
                    can_calibrate = (
                        len(patient_data_raw) >= 3 and
                        selected_marker in patient_data_raw.columns and
                        patient_data_raw[selected_marker].dropna().shape[0] >= 3
                    )
                    
                    calibrate_button_disabled = not can_calibrate
                    calibrate_help = (
                        "用您的历史数据校准模型参数（需10-30秒）"
                        if can_calibrate
                        else "需要至少3次该标志物的检测记录"
                    )
                    
                    # 校准按钮
                    if st.button(
                        "🔬 智能校准模型",
                        disabled=calibrate_button_disabled,
                        help=calibrate_help,
                        use_container_width=True,
                        type="primary" if not calibration_status else "secondary",
                        key="calibrate_model_button"  # ✅ 添加唯一key
                    ):
                        with st.spinner("🔬 正在校准模型（约10-30秒，请耐心等待）..."):
                            calibration = simulation_engine.calibrate_model_with_scipy(
                                model_name=selected_model,
                                patient_history=patient_data_raw,
                                selected_marker=selected_marker
                            )
                        
                        if calibration['success']:
                            # 保存校准结果
                            data_manager.save_calibrated_params(
                                patient_id,
                                selected_model,
                                selected_marker,
                                calibration
                            )
                            
                            if calibration['is_reliable']:
                                st.success(
                                    f"✅ **校准成功！**\n\n"
                                    f"**验证结果**：\n"
                                    f"- 验证误差: **{calibration['validation_error']:.1%}**\n"
                                    f"- 预测值: {calibration['predicted_value']:.2f}\n"
                                    f"- 实际值: {calibration['actual_value']:.2f}\n"
                                    f"- 验证日期: {calibration['validation_date']}\n\n"
                                    f"**优化信息**：\n"
                                    f"- 方法: {calibration['optimization_info']['method']}\n"
                                    f"- 迭代次数: {calibration['optimization_info']['iterations']}\n"
                                    f"- 是否收敛: {'✅ 是' if calibration['optimization_info']['convergence'] else '❌ 否'}\n\n"
                                    f"后续模拟将自动使用校准参数。"
                                )
                            else:
                                st.warning(
                                    f"⚠️ **校准完成，但误差较大**\n\n"
                                    f"- 验证误差: **{calibration['validation_error']:.1%}**（>25%）\n"
                                    f"- 预测值: {calibration['predicted_value']:.2f}\n"
                                    f"- 实际值: {calibration['actual_value']:.2f}\n\n"
                                    f"**可能原因**：\n"
                                    f"1. 数据点太少（当前 {len(patient_data_raw)} 个点）\n"
                                    f"2. 数据波动较大或存在异常值\n"
                                    f"3. 该模型可能不适合您的情况\n\n"
                                    f"**提示**：\n"
                                    f"- 积累更多数据后再校准\n"
                                    f"- 尝试其他机理模型\n"
                                    f"- 检查数据是否有录入错误\n\n"
                                    f"当前模拟仍使用文献默认参数。"
                                )
                        else:
                            st.error(
                                f"❌ **校准失败**: {calibration.get('error', '未知错误')}\n\n"
                                f"请检查：\n"
                                f"1. 数据是否包含异常值\n"
                                f"2. 标志物选择是否正确\n"
                                f"3. 查看控制台日志获取详细错误"
                            )
                    
                    # 删除校准参数按钮（如果存在校准参数）
                    if calibration_status:
                        if st.button(
                            "🗑️ 重置参数",
                            help="删除当前校准参数，恢复使用文献默认值",
                            use_container_width=True,
                            key="reset_calibration_button"  # ✅ 添加唯一key
                        ):
                            if data_manager.delete_calibrated_params(
                                patient_id, selected_model, selected_marker
                            ):
                                st.info("✅ 已重置为文献默认参数")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ 重置失败，请查看日志")

                # ========================================
                # 第 4️⃣ 部分：设置模拟方案
                # ========================================
                st.markdown("---")
                st.subheader(language_support.t("simulation_plan"))
                
                st.caption("💡 提示：可输入多个阶段用于数学模拟（如：4周标准剂量 → 2周0%用药输入假设 → 4周75%剂量假设），实际用药请遵医嘱")
                
                # 动态添加阶段
                if 'num_phases' not in st.session_state:
                    st.session_state.num_phases = 2  # 默认2个阶段
                
                # 初始化展开状态字典
                if 'phase_expanded' not in st.session_state:
                    st.session_state.phase_expanded = {}
                
                col_add, col_remove, col_spacer = st.columns([1, 1, 3])
                
                with col_add:
                    if st.button("➕ 添加阶段", key="add_phase_button"):
                        old_num = st.session_state.num_phases
                        st.session_state.num_phases += 1
                        # 新添加的阶段默认展开
                        st.session_state.phase_expanded[old_num] = True
                        st.rerun()
                
                with col_remove:
                    if st.button("➖ 删除阶段", key="remove_phase_button") and st.session_state.num_phases > 1:
                        # 删除最后一个阶段的展开状态
                        last_idx = st.session_state.num_phases - 1
                        if last_idx in st.session_state.phase_expanded:
                            del st.session_state.phase_expanded[last_idx]
                        st.session_state.num_phases -= 1
                        st.rerun()
                
                # 收集每个阶段的配置
                treatment_phases = []
                
                for i in range(st.session_state.num_phases):
                    # 确定展开状态：新添加的展开，前两个默认展开，其他默认折叠
                    is_expanded = st.session_state.phase_expanded.get(
                        i, 
                        i < 2  # 前两个默认展开
                    )
                    
                    with st.expander(f"阶段 {i+1}", expanded=is_expanded):
                        col_duration, col_dose = st.columns(2)
                        
                        with col_duration:
                            duration = st.number_input(
                                f"持续时间（天）",
                                min_value=1,
                                max_value=90,
                                value=28 if i == 0 else 14,
                                step=7,
                                key=f"duration_{i}"
                            )
                        
                        with col_dose:
                            dose_name = st.selectbox(
                                f"剂量水平",
                                options=list(config.DOSE_PRESETS.keys()),
                                index=1,  # 默认"100% 标准剂量"
                                key=f"dose_{i}"
                            )
                        
                        treatment_phases.append((duration, dose_name))
                
                # 显示方案摘要
                total_days = sum(d for d, _ in treatment_phases)
                st.info(f"📅 方案总时长: {total_days} 天 ({total_days/7:.1f} 周)")
                
                # ========================================
                # 第 5️⃣ 部分：运行模拟
                # ========================================
                st.markdown("---")
                st.subheader(language_support.t("run_simulation"))
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 计算最后一次化验的日期（天数）
                    if not patient_data_raw.empty:
                        days_since_first = (patient_data_raw.index[-1] - patient_data_raw.index[0]).days
                    else:
                        days_since_first = 0
                    
                    # 模拟时长 = max(方案时长) + 15天
                    recommended_duration = total_days + 15
                    
                    sim_duration = st.number_input(
                        "模拟时长（天）",
                        min_value=total_days,
                        max_value=365,
                        value=recommended_duration,
                        step=7,
                        help=f"默认模拟至最后化验后15天（{recommended_duration}天）"
                    )
                
                with col2:
                    if st.button(language_support.t("start_simulation"), type="primary", use_container_width=True, key="start_simulation_button"):
                        # 从 controller 获取患者上下文
                        sim_context = health_controller.get_simulation_context()
                        st.session_state.sim_context = sim_context

                        with st.spinner("🔬 正在运行机理模型..."):
                            try:
                                # 调用模拟函数
                                sim_result = simulation_engine.run_adaptive_simulation(
                                    model_name=selected_model,
                                    treatment_schedule=treatment_phases,
                                    initial_marker_value=sim_context['latest_values'][selected_marker],
                                    patient_history=sim_context['patient_data_raw'],
                                    simulation_days=sim_duration,
                                    selected_marker=selected_marker,
                                    patient_id=patient_id
                                )

                            except Exception as sim_e:
                                logger.error(f"模拟执行失败: {sim_e}", exc_info=True)
                                st.error(
                                    f"❌ 模拟过程中出错：{str(sim_e)}\n\n"
                                    f"请检查：\n"
                                    f"1. 初始标志物值是否合理（当前值：{sim_context['latest_values'][selected_marker]:.2f}）\n"
                                    f"2. 模拟方案参数是否合理\n"
                                    f"3. 查看控制台日志获取详细错误"
                                )
                                sim_result = {'success': False, 'error': str(sim_e)}
                        
                        # 显示结果
                        if sim_result.get('success'):
                            st.session_state.sim_result = sim_result
                            st.session_state.sim_marker = selected_marker
                            st.success("✅ 模拟完成！")
                            logger.info(
                                f"模拟成功完成 - 模型: {selected_model}, "
                                f"标志物: {selected_marker}, "
                                f"时长: {sim_duration}天"
                            )
                        else:
                            error_msg = sim_result.get('error', '未知错误')
                            st.error(f"❌ 模拟失败: {error_msg}")
                            logger.error(f"模拟失败: {error_msg}")

                # ========================================
                # 第 6️⃣ 部分：显示结果
                # ========================================
                if 'sim_result' in st.session_state and st.session_state.sim_result:
                    st.markdown("---")
                    st.subheader(language_support.t("simulation_results"))
                    
                    result = st.session_state.sim_result
                    marker_name = st.session_state.sim_marker
                    
                    # === 5.1 关键指标卡片 ===
                    col1, col2, col3, col4 = st.columns(4)
                    
                    initial_burden = result['total_burden'][0]
                    final_burden = result['total_burden'][-1]
                    min_burden = result['total_burden'].min()
                    time_to_min = result['time'][np.argmin(result['total_burden'])]
                    
                    with col1:
                        st.metric(
                            "初始值",
                            f"{initial_burden:.2f}",
                            help="模拟起始时的肿瘤负荷"
                        )
                    
                    with col2:
                        reduction = (initial_burden - min_burden) / initial_burden * 100
                        st.metric(
                            "最佳反应",
                            f"{min_burden:.2f}",
                            delta=f"-{reduction:.1f}%",
                            delta_color="inverse",
                            help="模拟期间达到的最低值"
                        )
                    
                    with col3:
                        st.metric(
                            "最佳反应时间",
                            f"{int(time_to_min)} 天",
                            help="达到最低值的时间"
                        )
                    
                    with col4:
                        final_change = (final_burden - initial_burden) / initial_burden * 100
                        st.metric(
                            "模拟终点值",
                            f"{final_burden:.2f}",
                            delta=f"{final_change:+.1f}%",
                            help="模拟结束时的肿瘤负荷"
                        )
                    
                    # === 5.2 主图：肿瘤负荷演化 ===
                    
                    fig = go.Figure()

                    uncertainty = result.get('uncertainty', {})
                    if uncertainty.get('available'):
                        p10 = np.asarray(uncertainty['p10'])
                        p90 = np.asarray(uncertainty['p90'])
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([result['time'], result['time'][::-1]]),
                            y=np.concatenate([p90, p10[::-1]]),
                            fill='toself',
                            fillcolor='rgba(231, 76, 60, 0.14)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            name='模型不确定性区间'
                        ))
                    
                    # 肿瘤负荷曲线
                    fig.add_trace(go.Scatter(
                        x=result['time'],
                        y=result['total_burden'],
                        mode='lines',
                        name='预测的肿瘤负荷',
                        line=dict(color='#e74c3c', width=3),
                        hovertemplate='第 %{x} 天<br>肿瘤负荷: %{y:.2f}<extra></extra>'
                    ))
                    
                    # 参考线：初始值
                    fig.add_hline(
                        y=initial_burden,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"初始值 ({initial_burden:.2f})",
                        annotation_position="right"
                    )
                    
                    # 参考线：正常上限
                    if ref_upper:
                        fig.add_hline(
                            y=ref_upper,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"正常上限 ({ref_upper:.2f})",
                            annotation_position="right"
                        )
                    
                    # 治疗阶段背景色
                    cumulative_time = 0
                    for duration, dose_name in treatment_phases:
                        intensity = config.DOSE_PRESETS[dose_name]
                        
                        if intensity > 0:
                            fig.add_vrect(
                                x0=cumulative_time,
                                x1=cumulative_time + duration,
                                fillcolor="rgba(52, 152, 219, 0.2)",
                                line_width=0,
                                annotation_text=dose_name,
                                annotation_position="top left"
                            )
                        
                        cumulative_time += duration
                    
                    # 标记最佳反应点
                    fig.add_trace(go.Scatter(
                        x=[time_to_min],
                        y=[min_burden],
                        mode='markers',
                        name='最佳反应点',
                        marker=dict(size=15, color='#2ecc71', symbol='star'),
                        hovertemplate=f'最佳反应<br>第 {int(time_to_min)} 天<br>{min_burden:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=dict(
                            text=f"{marker_name} 预测演化轨迹",
                            font=dict(size=18)
                        ),
                        xaxis=dict(title="时间（天）"),
                        yaxis=dict(title=f"{marker_name} 值"),
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # === 5.3 亚群动态图（如果适用）===
                    if len(result['state_names']) > 1:
                        st.markdown("#### 肿瘤亚群动态")
                        
                        fig_sub = go.Figure()
                        
                        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
                        
                        for i, state_name in enumerate(result['state_names']):
                            fig_sub.add_trace(go.Scatter(
                                x=result['time'],
                                y=result['states'][:, i],
                                mode='lines',
                                name=state_name,
                                line=dict(color=colors[i % len(colors)], width=2),
                                stackgroup='one'  # 堆叠面积图
                            ))
                        
                        fig_sub.update_layout(
                            title="肿瘤亚群演化（堆叠图）",
                            xaxis=dict(title="时间（天）"),
                            yaxis=dict(title="细胞数量"),
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_sub, use_container_width=True)
                    
                    # === 5.4 观察提示 ===
                    st.markdown("#### 💡 模拟观察提示")
                    
                    # 根据结果生成观察提示
                    if final_burden < initial_burden * 0.5:
                        advice_level = "success"
                        advice_icon = "✅"
                        advice_text = (
                            f"**曲线下降明显**：模拟显示输入方案下肿瘤负荷下降超过50%。"
                            f"请仅作为观察结果，实际治疗安排请咨询医生。"
                        )
                    elif final_burden < initial_burden * 0.8:
                        advice_level = "info"
                        advice_icon = "ℹ️"
                        advice_text = (
                            f"**曲线小幅下降**：模拟显示肿瘤负荷有所下降，但幅度有限。"
                            f"请不要据此自行调整剂量；如担心控制不足，请咨询医生。"
                        )
                    else:
                        advice_level = "warning"
                        advice_icon = "⚠️"
                        advice_text = (
                            f"**曲线控制有限**：模拟显示输入方案下肿瘤负荷下降不明显。"
                            f"可向医生咨询是否需要调整治疗策略或进一步检查。"
                        )
                    
                    if advice_level == "success":
                        st.success(f"{advice_icon} {advice_text}")
                    elif advice_level == "info":
                        st.info(f"{advice_icon} {advice_text}")
                    else:
                        st.warning(f"{advice_icon} {advice_text}")

                    host_context = result.get('host_context', {})
                    if host_context:
                        with st.expander("🧭 基础化验如何影响本次模拟", expanded=False):
                            host_scores = host_context.get('scores', {})
                            st.markdown(
                                f"- 炎症负荷代理：`{host_scores.get('inflammation_load', 0):.2f}`\n"
                                f"- 免疫储备代理：`{host_scores.get('immune_reserve', 0):.2f}`\n"
                                f"- 营养储备代理：`{host_scores.get('nutrition_reserve', 0):.2f}`\n"
                                f"- 数据约束强度：`{host_scores.get('data_confidence', 0):.2f}`"
                            )
                            for obs in host_context.get('observations', []):
                                st.caption(f"- {obs}")

                            uncertainty = result.get('uncertainty', {})
                            if uncertainty.get('available'):
                                st.caption(
                                    f"本次不确定性区间由 {uncertainty.get('sample_count', 0)} 条参数扰动轨迹生成；"
                                    f"基础化验越少、炎症代理越高，区间通常越宽。"
                                )

                    evidence_annotations = result.get('evidence_annotations', simulation_engine.get_evidence_annotations())
                    with st.expander("📚 论文注释与模型边界", expanded=False):
                        for note in evidence_annotations:
                            st.markdown(
                                f"**{note['title']}**  \n"
                                f"{note['citation']} DOI: [{note['doi']}]({note['url']})  \n"
                                f"- 本系统用途：{note['model_use']}  \n"
                                f"- 边界：{note['caution']}"
                            )
                    
                    st.caption("""
                    **📖 如何与医生讨论这个结果**：
                    1. 截图保存上方的演化曲线
                    2. 向医生说明："我用数学模型模拟了一个假设方案"
                    3. 重点展示"最佳反应时间"和"终点值"
                    4. 询问医生："实际临床中，这个预测合理吗？"
                    5. 如医生认为有意义，再调整模拟参数
                    """)
                    
                    # === 5.5 导出报告 ===
                    with st.expander("📄 导出模拟报告"):
                        report_data = {
                            '模型': selected_model,
                            '标志物': marker_name,
                            '初始值': initial_burden,
                            '最低值': min_burden,
                            '降幅': f"{reduction:.1f}%",
                            '最佳反应时间': f"{int(time_to_min)}天",
                            '终点值': final_burden,
                            '方案': ' → '.join([f"{d}天@{dose}" for d, dose in treatment_phases])
                        }
                        
                        st.json(report_data)
                        
                        # TODO: 添加导出为 PDF 的功能（需要额外的库）
                        st.info("💡 提示：您可以截图保存此页面，或使用浏览器的「打印」功能导出为PDF")


else:
    logger.info("未选择病人，显示欢迎界面。")
    st.info(language_support.t("welcome_select_patient"))
    st.markdown(language_support.t("welcome_title"))
    st.markdown(language_support.t("welcome_body"))
