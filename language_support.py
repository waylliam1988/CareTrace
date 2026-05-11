# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

"""Bilingual language support for CareTrace."""

from __future__ import annotations

import streamlit as st


LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "English",
}

LABEL_TO_CODE = {label: code for code, label in LANGUAGE_LABELS.items()}
DEFAULT_LANGUAGE = "zh"
LANGUAGE_STATE_KEY = "caretrace_language"
LANGUAGE_WIDGET_KEY = "caretrace_language_selector"


TRANSLATIONS: dict[str, dict[str, str]] = {
    "app_title": {
        "zh": "🩺 CareTrace 关照轨迹",
        "en": "🩺 CareTrace",
    },
    "language_label": {
        "zh": "语言 / Language",
        "en": "Language / 语言",
    },
    "patient_profile": {
        "zh": "👤 病人档案",
        "en": "👤 Patient Profile",
    },
    "select_patient": {
        "zh": "选择档案",
        "en": "Select profile",
    },
    "select_patient_placeholder": {
        "zh": "<选择病人>",
        "en": "<Select patient>",
    },
    "new_profile": {
        "zh": "➕ 新建档案",
        "en": "➕ New Profile",
    },
    "enter_name": {
        "zh": "输入姓名",
        "en": "Enter name",
    },
    "create_profile": {
        "zh": "确定创建",
        "en": "Create Profile",
    },
    "created": {
        "zh": "创建成功!",
        "en": "Created.",
    },
    "profile_exists": {
        "zh": "档案已存在",
        "en": "This profile already exists.",
    },
    "name_required": {
        "zh": "请输入姓名",
        "en": "Please enter a name.",
    },
    "tab_report": {
        "zh": "📊 健康报告",
        "en": "📊 Health Report",
    },
    "tab_entry": {
        "zh": "✏️ 录入数据",
        "en": "✏️ Data Entry",
    },
    "tab_history": {
        "zh": "📂 历史记录",
        "en": "📂 History",
    },
    "tab_simulation": {
        "zh": "🧪 适应性模拟",
        "en": "🧪 Adaptive Simulation",
    },
    "medical_disclaimer": {
        "zh": (
            "⚠️ **重要提示：CareTrace 不提供医疗建议**\n"
            "- 所有分析结果仅为数据统计观察和就诊沟通准备。\n"
            "- 它不能诊断复发、进展、疗效或毒副反应。\n"
            "- 任何治疗、停药、加药、检查安排都应咨询有资质的医生。"
        ),
        "en": (
            "⚠️ **Important: CareTrace does not provide medical advice**\n"
            "- All outputs are statistical observations for personal records and clinician discussions.\n"
            "- It does not diagnose recurrence, progression, treatment response, or toxicity.\n"
            "- Treatment changes, dosing, drug holds, and imaging or lab schedules must be discussed with qualified clinicians."
        ),
    },
    "no_health_data": {
        "zh": "📭 暂无健康数据。请在「✏️ 录入数据」中添加第一份检验报告。",
        "en": "📭 No health data yet. Add the first lab report in Data Entry.",
    },
    "latest_test": {
        "zh": "**最近检测**: {date} ({phase})",
        "en": "**Latest test**: {date} ({phase})",
    },
    "latest_missing": {
        "zh": "无法获取最近检测信息。",
        "en": "Unable to load latest test information.",
    },
    "analyzing_data": {
        "zh": "🔍 正在分析健康数据...",
        "en": "🔍 Analyzing health data...",
    },
    "report_error": {
        "zh": "生成健康报告时出错: {error}",
        "en": "Error while generating the health report: {error}",
    },
    "baseline_deviation": {
        "zh": "### 🚨 检测到个人基线显著偏离",
        "en": "### 🚨 Notable deviation from personal baseline",
    },
    "gauge_failed": {
        "zh": "仪表盘加载失败",
        "en": "Gauge failed to load.",
    },
    "current_value": {
        "zh": "**📍 当前值**: {value}",
        "en": "**📍 Current value**: {value}",
    },
    "personal_baseline": {
        "zh": "**📏 个人基线 (中位数)**: {value}",
        "en": "**📏 Personal baseline (median)**: {value}",
    },
    "deviation_degree": {
        "zh": "**📊 偏离程度**: {level}",
        "en": "**📊 Degree of deviation**: {level}",
    },
    "z_caption": {
        "zh": "(相比您的历史基线，统计值: {z_score:.1f}σ)",
        "en": "(Compared with your personal baseline: {z_score:.1f}σ)",
    },
    "interpretation": {
        "zh": "**💬 解读**",
        "en": "**💬 Interpretation**",
    },
    "observation_note": {
        "zh": "**💡 观察提示**",
        "en": "**💡 Observation Note**",
    },
    "no_interpretation": {
        "zh": "无解读信息",
        "en": "No interpretation available.",
    },
    "no_observation_note": {
        "zh": "暂无观察提示",
        "en": "No observation note available.",
    },
    "short_term_prediction": {
        "zh": "### 📈 短期趋势预测",
        "en": "### 📈 Short-Term Trend Forecast",
    },
    "select_indicator": {
        "zh": "#### 🔍 选择要查看的指标",
        "en": "#### 🔍 Select indicators to review",
    },
    "no_prediction": {
        "zh": "💡 暂无趋势预测结果。请确保已录入足够数据，并在下方点击「🔄 重新训练模型」。",
        "en": "💡 No trend forecast yet. Add enough data and click “Retrain Model” below.",
    },
    "choose_at_least_one_indicator": {
        "zh": "💡 请在下方选择至少一个指标查看预测",
        "en": "💡 Select at least one indicator below to view forecasts.",
    },
    "multi_select_hint": {
        "zh": "可多选，图表会以标签页形式展示",
        "en": "You can select multiple indicators. Charts will be shown as tabs.",
    },
    "remaining_indicators": {
        "zh": "💡 还有 {count} 个",
        "en": "💡 {count} more",
    },
    "all_selected": {
        "zh": "✅ 已选全部",
        "en": "✅ All selected",
    },
    "system_actions": {
        "zh": "⚙️ 系统操作",
        "en": "⚙️ System Actions",
    },
    "retrain_model": {
        "zh": "🔄 重新训练模型",
        "en": "🔄 Retrain Model",
    },
    "retrain_help": {
        "zh": "当您添加/修改数据或反馈后，点击此按钮让AI学习最新信息",
        "en": "Click after adding, editing, or labeling data so the model can learn the latest information.",
    },
    "clear_cache": {
        "zh": "🧹 清除内部缓存",
        "en": "🧹 Clear Internal Cache",
    },
    "clear_cache_help": {
        "zh": "强制清除所有内存缓存，下次访问将重新加载和计算",
        "en": "Clear in-memory cache. The next visit will reload and recompute data.",
    },
    "training_model": {
        "zh": "🤖 正在训练模型 (可能需要一些时间)...",
        "en": "🤖 Training models. This may take a while...",
    },
    "training_done": {
        "zh": "✅ 模型训练完成！报告已是最新状态。",
        "en": "✅ Model training complete. The report is up to date.",
    },
    "training_done_fallback": {
        "zh": "✅ 模型训练完成（默认权重模式）。请补充数据后再次训练以启用个性化分析。",
        "en": "✅ Training complete using default weights. Add more data and retrain to enable personalized analysis.",
    },
    "cache_cleared": {
        "zh": "✅ 所有内部缓存已清除！",
        "en": "✅ Internal cache cleared.",
    },
    "cache_clear_error": {
        "zh": "清除缓存时出错",
        "en": "Error while clearing cache.",
    },
    "data_entry_header": {
        "zh": "✏️ 录入新化验单",
        "en": "✏️ Add a New Lab Report",
    },
    "ocr_expander": {
        "zh": "📷 上传报告单图片自动识别（本地 OCR，保存前请核对）",
        "en": "📷 Upload Report Images for Local OCR (review before saving)",
    },
    "ocr_privacy": {
        "zh": "图片只在本机用 RapidOCR/OpenCV 识别；识别结果会先进入确认表，不会自动写入数据库。",
        "en": "Images are processed locally with RapidOCR/OpenCV. Results go to a review table first and are not saved automatically.",
    },
    "upload_report_images": {
        "zh": "上传报告单截图或照片",
        "en": "Upload report screenshots or photos",
    },
    "ocr_button": {
        "zh": "🔍 识别上传图片",
        "en": "🔍 Recognize Uploaded Images",
    },
    "ocr_running": {
        "zh": "正在本地识别图片...",
        "en": "Recognizing images locally...",
    },
    "ocr_date_label": {
        "zh": "检查日期",
        "en": "Test date",
    },
    "report_date": {
        "zh": "📅 检查日期",
        "en": "📅 Test date",
    },
    "phase_context": {
        "zh": "选择化验单的背景阶段",
        "en": "Choose the context for this lab report",
    },
    "phase_help": {
        "zh": "“稳定监控期”指在家休养、定期复查或口服维持药物期间。\n“强效治疗期”指正在住院、化疗、放疗、刚做完手术等主要治疗活动期间。",
        "en": "Stable monitoring: home recovery, routine follow-up, or maintenance oral medication.\nIntensive treatment: hospitalization, chemotherapy, radiotherapy, recent surgery, or major treatment activity.",
    },
    "manual_entry_hint": {
        "zh": "💡 请填写化验单数据，支持分次填写、切换模板不丢失",
        "en": "💡 Enter lab values here. You can fill them in batches and switch templates without losing staged data.",
    },
    "health_phase": {
        "zh": "##### 🏥 健康阶段",
        "en": "##### 🏥 Health Phase",
    },
    "fill_indicators": {
        "zh": "### 📝 填写指标数据",
        "en": "### 📝 Enter Lab Indicators",
    },
    "select_report_type": {
        "zh": "选择化验单类型",
        "en": "Select lab report type",
    },
    "value_column": {
        "zh": "填写数值",
        "en": "Value",
    },
    "indicator_column": {
        "zh": "指标",
        "en": "Indicator",
    },
    "raw_value_column": {
        "zh": "数值",
        "en": "Value",
    },
    "reference_range_column": {
        "zh": "正常范围",
        "en": "Reference range",
    },
    "unit_column": {
        "zh": "单位",
        "en": "Unit",
    },
    "value_help": {
        "zh": "填写后自动保存，可切换模板继续填写",
        "en": "Values are staged automatically; you can switch templates and continue.",
    },
    "template_switch_hint": {
        "zh": "💡 提示：可切换多个模板分次填写",
        "en": "💡 Tip: You can switch between templates and enter values in multiple passes.",
    },
    "filled_count": {
        "zh": "✅ 已填写 {count} 个指标",
        "en": "✅ {count} indicators entered",
    },
    "filled_preview": {
        "zh": "📋 查看已填写内容",
        "en": "📋 Review entered values",
    },
    "save_report": {
        "zh": "💾 保存化验单",
        "en": "💾 Save Lab Report",
    },
    "need_one_indicator": {
        "zh": "⚠️ 请先填写至少一个指标数据",
        "en": "⚠️ Please enter at least one indicator first.",
    },
    "save_success_count": {
        "zh": "✅ 成功保存 {count} 个指标!",
        "en": "✅ Saved {count} indicators.",
    },
    "clear_form": {
        "zh": "🗑️ 清空重填",
        "en": "🗑️ Clear Form",
    },
    "history_header": {
        "zh": "📂 历史记录",
        "en": "📂 History",
    },
    "history_records": {
        "zh": "查看记录",
        "en": "Records",
    },
    "reference_ranges": {
        "zh": "参考范围设置",
        "en": "Reference Ranges",
    },
    "expand_record_hint": {
        "zh": "点击展开任意记录进行编辑或删除",
        "en": "Expand any record to edit or delete it.",
    },
    "no_records": {
        "zh": "📋 暂无记录，请先录入数据",
        "en": "📋 No records yet. Please add data first.",
    },
    "edit": {
        "zh": "✏️ 修改",
        "en": "✏️ Edit",
    },
    "delete": {
        "zh": "🗑️ 删除",
        "en": "🗑️ Delete",
    },
    "deleted": {
        "zh": "已删除",
        "en": "Deleted.",
    },
    "reference_range_header": {
        "zh": "指标正常值范围",
        "en": "Reference Ranges",
    },
    "import_defaults": {
        "zh": "📥 从系统导入默认值",
        "en": "📥 Import System Defaults",
    },
    "import_success": {
        "zh": "导入成功!",
        "en": "Imported.",
    },
    "save_settings": {
        "zh": "💾 保存设置",
        "en": "💾 Save Settings",
    },
    "save_success": {
        "zh": "✅ 保存成功!",
        "en": "✅ Saved.",
    },
    "no_save_data": {
        "zh": "没有需要保存的数据",
        "en": "No data to save.",
    },
    "add_first_record": {
        "zh": "📭 请先在「✏️ 录入数据」中添加至少一次检测记录",
        "en": "📭 Please add at least one lab record in Data Entry first.",
    },
    "choose_model_help": {
        "zh": "不同模型对应不同的肿瘤生长假设和用药输入",
        "en": "Different models represent different tumor-growth assumptions and treatment inputs.",
    },
    "choose_model_select": {
        "zh": "选择要查看的模型假设",
        "en": "Choose a model assumption",
    },
    "simulation_header": {
        "zh": "🧪 治疗方案数学模拟器",
        "en": "🧪 Treatment Schedule Math Simulator",
    },
    "simulation_intro": {
        "zh": (
            "**功能说明**：\n"
            "- 基于肿瘤生长的机理模型，而不是简单直线外推。\n"
            "- 模拟不同输入方案下的理论肿瘤标志物轨迹。\n"
            "- 结合血常规、炎症和营养指标，生成更透明的不确定性区间。\n"
            "- 帮助整理可与医生讨论的用药阶段假设和复查问题。\n\n"
            "⚠️ **重要提醒**：\n"
            "- 这是数学模型模拟，不能替代医生的临床判断。\n"
            "- 实际疗效受耐药性、免疫状态、感染、近期治疗等多种因素影响。\n"
            "- 基础化验只能提供间接代理变量，不能推出个人剂量。\n"
            "- 请将模拟结果仅作为与医生讨论的观察材料。"
        ),
        "en": (
            "**What this simulator does**:\n"
            "- Uses mechanistic tumor-growth models rather than simple linear extrapolation.\n"
            "- Simulates theoretical tumor-marker trajectories under different input schedules.\n"
            "- Uses CBC, inflammation, and nutrition proxies to express uncertainty more honestly.\n"
            "- Helps prepare dosing-stage hypotheses and follow-up questions for clinician discussions.\n\n"
            "⚠️ **Important**:\n"
            "- This is a mathematical simulation, not a substitute for clinical judgment.\n"
            "- Real response depends on resistance, immune state, infection, recent treatment, and many other factors.\n"
            "- Basic lab reports can only provide indirect proxies; they cannot determine personal dosing.\n"
            "- Use the simulation only as discussion material for qualified clinicians."
        ),
    },
    "choose_model": {
        "zh": "1️⃣ 选择机理模型",
        "en": "1️⃣ Choose Mechanistic Model",
    },
    "initial_state": {
        "zh": "2️⃣ 设置初始状态",
        "en": "2️⃣ Set Initial State",
    },
    "personal_calibration": {
        "zh": "3️⃣ 模型个性化校准（可选）",
        "en": "3️⃣ Optional Personal Calibration",
    },
    "simulation_plan": {
        "zh": "4️⃣ 设置模拟方案",
        "en": "4️⃣ Set Simulation Plan",
    },
    "run_simulation": {
        "zh": "5️⃣ 运行模拟",
        "en": "5️⃣ Run Simulation",
    },
    "start_simulation": {
        "zh": "🚀 开始模拟",
        "en": "🚀 Start Simulation",
    },
    "simulation_results": {
        "zh": "📊 模拟结果",
        "en": "📊 Simulation Results",
    },
    "welcome_select_patient": {
        "zh": "👈 请在左侧侧边栏选择或创建一个病人档案以开始使用。",
        "en": "👈 Select or create a patient profile in the sidebar to begin.",
    },
    "welcome_title": {
        "zh": "### 欢迎使用 CareTrace 关照轨迹!",
        "en": "### Welcome to CareTrace.",
    },
    "welcome_body": {
        "zh": (
            "1. **直观洞察（健康报告）**：打开应用即可看到核心指标的未来趋势观察和健康数据提示。\n"
            "2. **轻松管理（数据管理中心）**：方便回顾、编辑历史化验单，并统一管理医学参考范围。\n"
            "3. **便捷录入（录入新化验单）**：使用智能模板或本地 OCR，尽量减少手动输入。\n\n"
            "**🔒 隐私安全**：您的所有数据都存储在您自己的电脑上，不会上传到任何服务器。"
        ),
        "en": (
            "1. **Health Report**: review key trends and data observations when the app opens.\n"
            "2. **Data Management**: review and edit historical lab reports and reference ranges.\n"
            "3. **Easy Entry**: use templates or local OCR to reduce manual typing.\n\n"
            "**🔒 Privacy**: all data stay on your own computer and are not uploaded to any server."
        ),
    },
}


PHASE_TRANSLATIONS = {
    "稳定监控期": {"zh": "稳定监控期", "en": "Stable monitoring"},
    "强效治疗期": {"zh": "强效治疗期", "en": "Intensive treatment"},
    "治疗调整期": {"zh": "治疗调整期", "en": "Treatment adjustment"},
    "恢复观察期": {"zh": "恢复观察期", "en": "Recovery observation"},
}


DEVIATION_TRANSLATIONS = {
    "严重偏离": {"zh": "严重偏离", "en": "marked deviation"},
    "中度偏离": {"zh": "中度偏离", "en": "moderate deviation"},
    "轻度偏离": {"zh": "轻度偏离", "en": "mild deviation"},
    "正常范围": {"zh": "正常范围", "en": "within usual range"},
}


def init_language() -> str:
    widget_value = st.session_state.get(LANGUAGE_WIDGET_KEY)
    if widget_value in LABEL_TO_CODE:
        st.session_state[LANGUAGE_STATE_KEY] = LABEL_TO_CODE[widget_value]
    elif LANGUAGE_STATE_KEY not in st.session_state:
        legacy_language = st.session_state.get("language")
        st.session_state[LANGUAGE_STATE_KEY] = (
            legacy_language if legacy_language in LANGUAGE_LABELS else DEFAULT_LANGUAGE
        )
    return st.session_state[LANGUAGE_STATE_KEY]


def get_language() -> str:
    return st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)


def _sync_language_from_widget() -> None:
    selected_label = st.session_state.get(LANGUAGE_WIDGET_KEY)
    st.session_state[LANGUAGE_STATE_KEY] = LABEL_TO_CODE.get(selected_label, DEFAULT_LANGUAGE)


def language_selector() -> str:
    current = init_language()
    labels = list(LABEL_TO_CODE.keys())
    current_label = LANGUAGE_LABELS.get(current, LANGUAGE_LABELS[DEFAULT_LANGUAGE])
    if LANGUAGE_WIDGET_KEY not in st.session_state:
        st.session_state[LANGUAGE_WIDGET_KEY] = current_label
    selected_label = st.sidebar.selectbox(
        TRANSLATIONS["language_label"].get(current, TRANSLATIONS["language_label"]["zh"]),
        labels,
        index=labels.index(current_label),
        key=LANGUAGE_WIDGET_KEY,
        on_change=_sync_language_from_widget,
    )
    st.session_state[LANGUAGE_STATE_KEY] = LABEL_TO_CODE.get(selected_label, DEFAULT_LANGUAGE)
    return st.session_state[LANGUAGE_STATE_KEY]


def t(key: str, **kwargs) -> str:
    language = get_language()
    item = TRANSLATIONS.get(key, {})
    text = item.get(language) or item.get(DEFAULT_LANGUAGE) or key
    if kwargs:
        return text.format(**kwargs)
    return text


def phase_label(value: str) -> str:
    language = get_language()
    return PHASE_TRANSLATIONS.get(value, {}).get(language, value)


def deviation_label(value: str) -> str:
    language = get_language()
    return DEVIATION_TRANSLATIONS.get(value, {}).get(language, value)
