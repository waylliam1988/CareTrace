# data_manager.py
import os
import uuid
import json
import pickle
import sqlite3
import pandas as pd
from datetime import datetime

# 从全局配置中导入常量
import config
import tumor_models  # 需要导入模型配置

import logging
logger = logging.getLogger(__name__)

# --- 模块初始化：确保模型存储目录存在 ---
try:
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    logger.debug(f"模型目录 '{config.MODELS_DIR}' 已确认存在。")
except OSError as e:
    logger.error(f"创建模型目录 '{config.MODELS_DIR}' 失败: {e}", exc_info=True)


# --- 数据库连接管理 ---

def get_db_connection():
    """建立并返回一个到SQLite数据库的连接。"""
    logger.debug(f"正在连接到数据库: {config.DB_FILE}")
    try:
        conn = sqlite3.connect(config.DB_FILE)
        return conn
    except sqlite3.Error as e:
        logger.error(f"无法连接到数据库 {config.DB_FILE}: {e}", exc_info=True)
        raise # 重新引发异常，让调用者知道连接失败


def init_db():
    """
    初始化数据库。如果相关的表不存在，则创建它们。
    使用 'with' 语句确保连接的自动管理和事务的完整性。
    """
    logger.info("正在初始化数据库...")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()

            logger.debug("正在创建 'patients' 表 (如果不存在)...")
            # 创建病人表
            c.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                )
            ''')

            logger.debug("正在创建 'lab_reports' 表 (如果不存在)...")
            # 创建化验单记录表
            c.execute('''
                CREATE TABLE IF NOT EXISTS lab_reports (
                    report_uuid TEXT NOT NULL,
                    patient_id INTEGER,
                    report_date TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK(phase IN ('强效治疗期', '稳定监控期')),
                    item_name TEXT NOT NULL,
                    item_value REAL NOT NULL,
                    user_label TEXT CHECK(user_label IS NULL OR user_label IN ('benign', 'significant', 'lab_error')),
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    PRIMARY KEY (report_uuid, item_name)
                )
            ''')

            logger.debug("正在创建 'item_references' 表 (如果不存在)...")
            # 创建指标参考范围表
            c.execute('''
                CREATE TABLE IF NOT EXISTS item_references (
                    item_name TEXT PRIMARY KEY,
                    lower_bound REAL,
                    upper_bound REAL
                )
            ''')

            logger.debug("正在创建 'mogp_predictions' 表 (如果不存在)...")
            # 创建MOGP预测结果表
            c.execute('''
                CREATE TABLE IF NOT EXISTS mogp_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    results TEXT NOT NULL,
                    target_indicators TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')

            logger.debug("正在创建 'similarity_feedback' 表 (如果不存在)...")
            # 创建相似度反馈表
            c.execute('''
                CREATE TABLE IF NOT EXISTS similarity_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_uuid TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    is_similar INTEGER NOT NULL,
                    feedback_date TEXT NOT NULL,
                    UNIQUE(current_uuid, indicator)
                )
            ''')

            logger.debug("正在创建 'feedback_with_shap' 表 (如果不存在)...")
            c.execute('''
                CREATE TABLE IF NOT EXISTS feedback_with_shap (
                    observation_uuid TEXT PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    report_uuid TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('benign', 'significant', 'lab_error')),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    shap_values TEXT NOT NULL,
                    pattern_type TEXT,
                    shap_type TEXT NOT NULL DEFAULT 'proxy' CHECK(shap_type IN ('real', 'proxy')),
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
                    FOREIGN KEY (report_uuid) REFERENCES lab_reports (report_uuid) ON DELETE CASCADE
                )
            ''')

            conn.commit()
            logger.info("数据库初始化完成。所有表均已确认存在。")
    except sqlite3.Error as e:
        logger.error(f"数据库初始化期间发生错误: {e}", exc_info=True)



# --- 病人数据管理 ---

def get_patients():
    """从数据库中获取所有病人的列表。"""
    logger.debug("正在从数据库获取病人列表...")
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM patients", conn)
        logger.debug(f"成功获取到 {len(df)} 个病人。")
        return df
    except sqlite3.DatabaseError as e:
        logger.error(f"获取病人列表失败: {e}", exc_info=True)
        return pd.DataFrame(columns=['id', 'name']) # 返回空


def add_patient(name):
    """
    添加一个新病人到数据库。
    :param name: 病人姓名。
    :return: 成功返回 True，如果病人已存在则返回 False。
    """
    logger.info(f"尝试添加新病人: '{name}'")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO patients (name) VALUES (?)", (name,))
            conn.commit()
            logger.info(f"成功添加病人: '{name}' (ID: {c.lastrowid})")
        return True
    except sqlite3.IntegrityError:
        # 这是一个预期的“错误”（用户添加了重名），所以使用 WARNING
        logger.warning(f"添加病人失败: 名称 '{name}' 已存在。")
        return False
    except sqlite3.Error as e:
        logger.error(f"添加病人 '{name}' 时发生数据库错误: {e}", exc_info=True)
        return False


# --- 化验单数据管理 ---

def save_lab_report(patient_id: int, report_date: str, phase: str, items_df: pd.DataFrame):
    """
    将一份新的化验单数据批量保存到数据库。
    使用 executemany 提高插入效率。
    """
    logger.info(f"准备保存新化验单: PatientID={patient_id}, Date={report_date}, Phase={phase}")
    if items_df.empty:
        logger.error("保存化验单失败：传入的DataFrame为空。")
        raise ValueError("传入的DataFrame为空")

    required_columns = ['指标名称', '检测值']
    missing_columns = [col for col in required_columns if col not in items_df.columns]
    if missing_columns:
        logger.error(f"保存化验单失败：缺少必需的列: {missing_columns}, 当前列: {items_df.columns.tolist()}")
        raise ValueError(f"缺少必需的列: {missing_columns}, 当前列: {items_df.columns.tolist()}")

    report_uuid = str(uuid.uuid4())
    logger.debug(f"生成新 Report UUID: {report_uuid}")

    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            records_to_insert = [
                (report_uuid, patient_id, report_date, phase, row['指标名称'], float(row['检测值']))
                for _, row in items_df.iterrows()
            ]

            # 替换 print
            logger.debug(f"准备插入 {len(records_to_insert)} 条记录 (UUID: {report_uuid})")
            logger.debug(f"示例数据: {records_to_insert[0] if records_to_insert else '无'}")

            c.executemany(
                "INSERT INTO lab_reports (report_uuid, patient_id, report_date, phase, item_name, item_value) VALUES (?, ?, ?, ?, ?, ?)",
                records_to_insert
            )
            conn.commit()
            # 替换 print
            logger.info(f"✅ 成功插入 {c.rowcount} 条记录 (UUID: {report_uuid})")
            
    except sqlite3.Error as e:
        logger.error(f"保存化验单 {report_uuid} 时发生数据库错误: {e}", exc_info=True)
        conn.rollback() # 确保回滚
        raise # 重新引发异常，让调用者知道失败了


def delete_lab_report(report_uuid: str):
    """根据化验单的唯一ID(UUID)删除所有相关记录。"""
    logger.info(f"准备删除化验单: UUID={report_uuid}")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM lab_reports WHERE report_uuid = ?", (report_uuid,))
            conn.commit()
            logger.info(f"成功删除化验单 {report_uuid}。受影响行数: {c.rowcount}")
            if c.rowcount == 0:
                logger.warning(f"删除 {report_uuid} 时未找到匹配记录。")
    except sqlite3.Error as e:
        logger.error(f"删除化验单 {report_uuid} 时发生数据库错误: {e}", exc_info=True)
        conn.rollback()
        raise


def update_lab_report(patient_id: int, report_uuid: str, report_date: str, phase: str, items_df: pd.DataFrame):
    """
    【V8.1 修复】更新化验单，智能保留历史指标（防止数据丢失）
    
    核心改进：
    1. 先读取旧数据，再合并新数据（增量更新）
    2. 对于编辑器中未出现的指标 → 自动保留旧值
    3. 对于编辑器中的 None 值 → 自动保留旧值（表示未修改）
    4. 只有用户显式删除时才删除指标（需要在编辑器中添加删除按钮）
    
    【重要】此版本假设：
    - 如果指标在编辑器中缺失 → 保留旧值（用户未加载该指标）
    - 如果指标值为 None → 保留旧值（用户未修改）
    - 删除操作需要单独的 UI 机制（如删除按钮）
    """
    logger.info(f"准备更新化验单: UUID={report_uuid}, PatientID={patient_id}, Date={report_date}")
    
    if items_df.empty:
        logger.error("update_lab_report: items_df 为空，无法更新。")
        return
    
    required_columns = ['指标名称', '检测值']
    missing_columns = [col for col in required_columns if col not in items_df.columns]
    if missing_columns:
        logger.error(f"update_lab_report: items_df 缺少必要的列: {missing_columns}")
        return
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # 步骤 1：读取旧数据
            logger.debug(f"正在读取 {report_uuid} 的旧记录...")
            c.execute(
                "SELECT item_name, item_value FROM lab_reports WHERE report_uuid = ?",
                (report_uuid,)
            )
            old_records = {row[0]: row[1] for row in c.fetchall()}
            logger.info(f"  旧记录数：{len(old_records)} 个指标")
            
            #  步骤 2：清除反馈（数据变更后需重新标注）
            logger.debug(f"正在清除 {report_uuid} 的相关反馈标签...")
            c.execute("DELETE FROM feedback_with_shap WHERE report_uuid = ?", (report_uuid,))
            
            # 步骤 3：删除旧化验数据
            logger.debug(f"正在删除 {report_uuid} 的旧记录...")
            c.execute("DELETE FROM lab_reports WHERE report_uuid = ?", (report_uuid,))
            
            # 步骤 4：智能合并新旧数据
            merged_records = old_records.copy()  # 先复制所有旧数据（保护网）
            logger.debug(f"开始合并新数据（编辑器中共 {len(items_df)} 个指标）...")
            
            update_count = 0
            add_count = 0
            keep_count = 0  # 保留（未修改）的计数
            
            # 创建编辑器中指标的集合（用于快速查找）
            edited_indicators = set(items_df['指标名称'])
            
            for _, row in items_df.iterrows():
                indicator_name = row['指标名称']
                new_value = row['检测值']
                
                # 情况 1：新值有效 → 覆盖/新增
                if pd.notna(new_value):
                    if indicator_name in merged_records:
                        # 检查值是否真的改变了
                        old_value = merged_records[indicator_name]
                        if abs(float(new_value) - old_value) > 1e-6:
                            update_count += 1
                            logger.debug(f"  更新: {indicator_name} = {old_value:.2f} → {new_value}")
                        else:
                            keep_count += 1
                            logger.debug(f"  保留: {indicator_name} = {new_value} (未变)")
                    else:
                        add_count += 1
                        logger.debug(f"  新增: {indicator_name} = {new_value}")
                    
                    merged_records[indicator_name] = float(new_value)
                
                # 情况 2：新值为 None → 保留旧值（用户未修改）
                else:
                    if indicator_name in merged_records:
                        keep_count += 1
                        logger.debug(f"  保留: {indicator_name} = {merged_records[indicator_name]} (编辑器中为None)")
                    else:
                        # 编辑器中的 None，且旧数据中也没有 → 不插入
                        logger.debug(f"  跳过: {indicator_name}（新旧均为空）")
            
            # 步骤 5：保留编辑器中未出现的旧指标 
            old_not_in_editor = set(old_records.keys()) - edited_indicators
            if old_not_in_editor:
                logger.info(f"  自动保留 {len(old_not_in_editor)} 个未在编辑器中出现的旧指标:")
                for indicator in list(old_not_in_editor)[:5]:  # 只打印前5个
                    logger.debug(f"    - {indicator} = {old_records[indicator]}")
                if len(old_not_in_editor) > 5:
                    logger.debug(f"    ... 以及其他 {len(old_not_in_editor) - 5} 个指标")
            
            # 步骤 6：插入合并后的完整数据
            records_to_insert = [
                (report_uuid, patient_id, report_date, phase, name, value)
                for name, value in merged_records.items()
            ]
            
            logger.info(
                f"📊 数据合并完成：\n"
                f"  - 旧记录: {len(old_records)} 个\n"
                f"  - 编辑器: {len(items_df)} 个\n"
                f"  - 更新: {update_count} 个\n"
                f"  - 新增: {add_count} 个\n"
                f"  - 保留: {keep_count} 个 (包括编辑器中未变/为None的)\n"
                f"  - 自动保留: {len(old_not_in_editor)} 个 (编辑器未加载)\n"
                f"  - 最终插入: {len(records_to_insert)} 个"
            )
            
            c.executemany(
                "INSERT INTO lab_reports (report_uuid, patient_id, report_date, phase, item_name, item_value) VALUES (?, ?, ?, ?, ?, ?)",
                records_to_insert
            )
            
            conn.commit()
            logger.info(f"✅ 成功更新化验单 {report_uuid}（已自动保留所有历史指标）")
            
    except sqlite3.Error as e:
        logger.error(f"更新化验单 {report_uuid} 时发生数据库错误: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise




def save_or_merge_lab_report(patient_id: int, report_date: str, phase: str, items_df: pd.DataFrame):
    """
    【智能保存/合并】
    保存一份化验单。如果当天已有记录，则将新项目合并进去（或覆盖旧项目）；
    如果当天没有记录，则创建新记录。
    这依赖于 (report_uuid, item_name) 上的 PRIMARY KEY。
    """
    logger.info(f"准备 智能保存/合并 化验单: PatientID={patient_id}, Date={report_date}")
    if items_df.empty:
        logger.error("智能保存/合并失败：传入的DataFrame为空。")
        raise ValueError("传入的DataFrame为空")

    required_columns = ['指标名称', '检测值']
    missing_columns = [col for col in required_columns if col not in items_df.columns]
    if missing_columns:
        logger.error(f"智能保存/合并失败：缺少必需的列: {missing_columns}")
        raise ValueError(f"缺少必需的列: {missing_columns}")

    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            logger.debug(f"检查 {patient_id} 在 {report_date} 是否存在记录...")
            c.execute(
                "SELECT report_uuid FROM lab_reports WHERE patient_id = ? AND report_date = ? LIMIT 1",
                (patient_id, report_date)
            )
            existing = c.fetchone()
            
            if existing:
                report_uuid = existing[0]
                logger.info(f"检测到现有UUID: {report_uuid}，正在合并/覆盖数据...")
            else:
                report_uuid = str(uuid.uuid4())
                logger.info(f"未找到当天记录，创建新UUID: {report_uuid}")
                
            records_to_upsert = [
                (report_uuid, patient_id, report_date, phase, row['指标名称'], float(row['检测值']))
                for _, row in items_df.iterrows()
            ]
            
            logger.debug(f"准备为 {report_uuid} 插入/替换 {len(records_to_upsert)} 条记录...")
            c.executemany(
                "INSERT OR REPLACE INTO lab_reports (report_uuid, patient_id, report_date, phase, item_name, item_value) VALUES (?, ?, ?, ?, ?, ?)",
                records_to_upsert
            )
            conn.commit()
            logger.info(f"✅ 成功 插入/替换 {c.rowcount} 条记录 (UUID: {report_uuid})")

    except sqlite3.Error as e:
        logger.error(f"智能保存/合并 {report_uuid} 时发生数据库错误: {e}", exc_info=True)
        conn.rollback()
        raise



def load_patient_data(patient_id: int) -> pd.DataFrame:
    """
    加载指定病人的所有化验单数据 (包含 phase 和 user_label)
    
    【V7.2 改进】user_label 从 feedback_with_shap 表读取（优先级更高）
    
    :param patient_id: 病人的唯一ID。
    :return: 一个以报告日期为索引的宽格式DataFrame。
    """
    logger.info(f"正在加载 PatientID={patient_id} 的所有化验单数据...")
    try:
        with get_db_connection() as conn:
            # 1. 加载基础数据
            query = """
                SELECT report_uuid, report_date, phase, item_name, item_value
                FROM lab_reports 
                WHERE patient_id = ?
            """
            df = pd.read_sql_query(query, conn, params=(patient_id,))

            if df.empty:
                logger.warning(f"未找到 PatientID={patient_id} 的任何化验单数据。")
                return pd.DataFrame()

            logger.debug(f"从数据库加载了 {len(df)} 行原始数据，准备透视...")

            # 2. 透视数值
            df_pivot = df.pivot_table(
                index=['report_uuid', 'report_date'], 
                columns='item_name', 
                values='item_value'
            )

            # 3. 处理元数据 (phase)
            meta_df = df[['report_uuid', 'report_date', 'phase']].drop_duplicates(
                subset=['report_uuid']
            ).set_index(['report_uuid', 'report_date'])

            # 4. 从 feedback_with_shap 表读取最新标签
            label_query = """
                SELECT report_uuid, label
                FROM feedback_with_shap
                WHERE patient_id = ?
                  AND (report_uuid, timestamp) IN (
                      SELECT report_uuid, MAX(timestamp)
                      FROM feedback_with_shap
                      WHERE patient_id = ?
                      GROUP BY report_uuid
                  )
            """
            label_df = pd.read_sql_query(
                label_query, 
                conn, 
                params=(patient_id, patient_id)
            )
            
            # 将标签列重命名为 user_label（保持兼容性）
            if not label_df.empty:
                label_df = label_df.rename(columns={'label': 'user_label'})
                label_df = label_df.set_index('report_uuid')
                logger.debug(f"从 feedback_with_shap 加载了 {len(label_df)} 个标签")
            else:
                logger.debug("feedback_with_shap 表中暂无标签数据")
                label_df = pd.DataFrame(columns=['user_label'])
                label_df.index.name = 'report_uuid'

            # 5. 合并所有元数据
            df_pivot = df_pivot.join(meta_df, how='left')
            
            # 合并标签（先重置索引以便按 report_uuid 匹配）
            df_pivot = df_pivot.reset_index()
            df_pivot = df_pivot.merge(
                label_df, 
                on='report_uuid', 
                how='left'
            )

            # 6. 恢复索引
            df_pivot['report_date'] = pd.to_datetime(df_pivot['report_date'])
            df_pivot = df_pivot.sort_values(by='report_date').set_index('report_date')

            logger.info(f"成功加载并转换数据。最终 Shape: {df_pivot.shape}")
            return df_pivot

    except sqlite3.DatabaseError as e:
        logger.error(f"加载 PatientID={patient_id} 数据时发生数据库错误: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"透视 PatientID={patient_id} 数据时发生意外错误: {e}", exc_info=True)
        return pd.DataFrame()


# --- 模型与权重文件管理 ---

def save_model_and_weights(patient_id: int, models_dict: dict, weights: pd.Series, feature_columns: list = None):
    """将训练好的模型字典、权重和特征列表序列化并保存到文件。"""
    logger.info(f"准备保存模型和权重: PatientID={patient_id}")
    try:
        # 保存模型字典
        model_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_models.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(models_dict, f)
        logger.debug(f"模型字典已保存到: {model_path}")
            
        # 保存权重
        if weights is not None:
            weights_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_weights.json")
            weights.to_json(weights_path)
            logger.debug(f"权重已保存到: {weights_path}")
        else:
            # 添加日志明确说明跳过了权重保存
            logger.warning(f"(PID: {patient_id}) 权重为 None，跳过权重文件保存。")
        
        # 保存特征列表
        if feature_columns:
            features_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_features.json")
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(feature_columns, f, ensure_ascii=False)
            logger.debug(f"特征列表已保存到: {features_path}")
        
        logger.info(f"✅ 成功保存所有模型/权重文件: PatientID={patient_id}")

    except (IOError, pickle.PicklingError, Exception) as e:
        logger.error(f"保存模型/权重时失败: PatientID={patient_id}: {e}", exc_info=True)
        raise


def load_model_and_weights(patient_id: int):
    """从文件加载指定病人的模型字典、权重和特征列表。"""
    logger.info(f"准备加载模型和权重: PatientID={patient_id}")
    models_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_models.pkl")
    weights_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_weights.json")
    features_path = os.path.join(config.MODELS_DIR, f"patient_{patient_id}_features.json")
    
    models_dict, weights, feature_columns = None, None, None

    if os.path.exists(models_path) and os.path.exists(weights_path):
        logger.debug(f"找到模型和权重文件: {models_path}, {weights_path}")
        try:
            with open(models_path, 'rb') as f:
                models_dict = pickle.load(f)
            logger.debug("模型字典 (.pkl) 加载成功。")
            
            weights = pd.read_json(weights_path, typ='series')
            logger.debug("权重 (.json) 加载成功。")
            
            if os.path.exists(features_path):
                with open(features_path, 'r', encoding='utf-8') as f:
                    feature_columns = json.load(f)
                logger.debug("特征列表 (.json) 加载成功。")
            else:
                logger.warning(f"未找到特征列表文件: {features_path}")
                    
        except (EOFError, pickle.UnpicklingError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"加载模型/权重失败: PatientID={patient_id}。文件可能已损坏: {e}", exc_info=True)
            return None, None, None
        
        logger.info(f"✅ 成功加载模型/权重: PatientID={patient_id}")
    else:
        logger.warning(f"未找到模型/权重文件: PatientID={patient_id}。模型/权重文件不存在。")
        
    return models_dict, weights, feature_columns


# --- 指标参考范围管理 ---

def load_references() -> pd.DataFrame:
    """
    【V2.0 - 智能回退版】
    从数据库加载指标参考范围，如果数据库中没有，则回退到 config 中的默认值
    
    优先级：
    1. 数据库中用户自定义的范围（优先）
    2. config.LAB_REPORT_CONFIG 中的默认范围（回退）
    
    :return: DataFrame with columns ['lower_bound', 'upper_bound']
    """
    logger.debug("正在从数据库加载指标参考范围...")
    
    # === 步骤 1：从数据库加载用户自定义范围 ===
    try:
        with get_db_connection() as conn:
            db_refs = pd.read_sql_query(
                "SELECT * FROM item_references", 
                conn, 
                index_col='item_name'
            )
        logger.debug(f"数据库中找到 {len(db_refs)} 条用户自定义参考范围")
    except Exception as e:
        logger.warning(f"加载数据库参考范围失败: {e}", exc_info=True)
        db_refs = pd.DataFrame(columns=['lower_bound', 'upper_bound'])
    
    # === 步骤 2：从 config 提取所有默认参考范围 ===
    config_refs = {}
    for template_name, items in config.LAB_REPORT_CONFIG.items():
        for item in items:
            indicator_name = item['name']
            lower = item.get('lower')
            upper = item.get('upper')
            
            # 只添加至少有一个边界的指标
            if lower is not None or upper is not None:
                config_refs[indicator_name] = {
                    'lower_bound': lower,
                    'upper_bound': upper
                }
    
    logger.debug(f"config 中找到 {len(config_refs)} 个默认参考范围")
    
    # === 步骤 3：合并（数据库优先）===
    merged_refs = pd.DataFrame.from_dict(config_refs, orient='index')
    
    # 用数据库中的值覆盖默认值
    for indicator in db_refs.index:
        merged_refs.loc[indicator] = db_refs.loc[indicator]
    
    logger.info(
        f"✅ 参考范围加载完成：\n"
        f"  - 数据库自定义: {len(db_refs)} 项\n"
        f"  - config 默认: {len(config_refs)} 项\n"
        f"  - 最终返回: {len(merged_refs)} 项"
    )
    
    return merged_refs


def save_or_update_references(references_df: pd.DataFrame):
    """
    批量保存或更新指标的参考范围。
    (函数文档保持不变)
    """
    if references_df.empty:
        logger.warning("尝试保存参考范围，但传入的DataFrame为空。")
        return
        
    logger.info(f"准备 插入/替换 {len(references_df)} 条指标参考范围...")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            records_to_upsert = [
                (index, row['lower_bound'], row['upper_bound'])
                for index, row in references_df.iterrows()
            ]
            c.executemany(
                "INSERT OR REPLACE INTO item_references (item_name, lower_bound, upper_bound) VALUES (?, ?, ?)",
                records_to_upsert
            )
            conn.commit()
            logger.info(f"✅ 成功 插入/替换 {c.rowcount} 条指标参考范围。")
    except sqlite3.Error as e:
        logger.error(f"保存指标参考范围时发生数据库错误: {e}", exc_info=True)
        conn.rollback()
        raise


def save_mogp_results(patient_id: int, mogp_results: dict, target_indicators: list):
    """保存MOGP预测结果到数据库"""
    logger.info(f"准备保存 MOGP 结果: PatientID={patient_id}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        logger.debug("正在序列化 MOGP 结果为 JSON...")
        serializable_results = {}
        for indicator, data in mogp_results.items():
            serializable_results[indicator] = {
                'historical_dates': [d.isoformat() for d in data.get('historical_dates', [])],
                'historical_values': [float(v) for v in data.get('historical_values', [])],
                
                # 确保所有绘图键都被保存
                'future_dates': [d.isoformat() for d in data.get('future_dates', [])],
                'predicted_mean': [float(v) for v in data.get('predicted_mean', [])],  # <--- 新增
                'confidence_lower': [float(v) for v in data.get('confidence_lower', [])],
                'confidence_upper': [float(v) for v in data.get('confidence_upper', [])],
                
                'all_dates': [d.isoformat() for d in data.get('all_dates', [])],
                'all_predicted_mean': [float(v) for v in data.get('all_predicted_mean', [])],
                'all_uncertainty': [float(v) for v in data.get('all_uncertainty', [])],
                'all_trend': [float(v) for v in data.get('all_trend', [])],

                'next_check_days': data.get('next_check_days'),
                'confidence': data.get('confidence', 'medium'),
                'warning': data.get('warning')
            }
        
        results_json = json.dumps(serializable_results)
        indicators_json = json.dumps(target_indicators)
        last_updated = datetime.now().isoformat()
        
        cursor.execute("SELECT id FROM mogp_predictions WHERE patient_id = ?", (patient_id,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
                UPDATE mogp_predictions 
                SET results = ?, target_indicators = ?, last_updated = ?
                WHERE patient_id = ?
            """, (results_json, indicators_json, last_updated, patient_id))
        else:
            cursor.execute("""
                INSERT INTO mogp_predictions (patient_id, results, target_indicators, last_updated)
                VALUES (?, ?, ?, ?)
            """, (patient_id, results_json, indicators_json, last_updated))
        
        conn.commit()
        logger.info(f"✅ MOGP结果已保存 (病人ID: {patient_id})")
        
    except Exception as e:
        logger.error(f"❌ 保存MOGP结果失败: PatientID={patient_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def load_mogp_results(patient_id: int) -> tuple:
    """
    从数据库加载MOGP预测结果（带兼容性处理）
    
    :return: (mogp_results, target_indicators, last_updated) 或 (None, None, None)
    """
    logger.info(f"准备加载 MOGP 结果: PatientID={patient_id}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT results, target_indicators, last_updated 
            FROM mogp_predictions 
            WHERE patient_id = ?
        """, (patient_id,))
        
        row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"未在数据库中找到 MOGP 结果: PatientID={patient_id}")
            return None, None, None
        
        results_json, indicators_json, last_updated_str = row
        logger.debug("找到 MOGP 记录，正在反序列化...")
        
        serializable_results = json.loads(results_json)
        target_indicators = json.loads(indicators_json)
        last_updated = datetime.fromisoformat(last_updated_str)
        
        logger.debug("正在将日期字符串转换回 datetime 对象...")
        mogp_results = {}
        for indicator, data in serializable_results.items():
            # 【兼容性修复】如果旧数据缺少 predicted_mean，尝试从 all_predicted_mean 截取
            predicted_mean = data.get('predicted_mean')
            if not predicted_mean:
                logger.warning(f"⚠️ 旧数据缺少 'predicted_mean'，尝试从 'all_predicted_mean' 生成")
                all_mean = data.get('all_predicted_mean', [])
                hist_len = len(data.get('historical_dates', []))
                predicted_mean = all_mean[hist_len:] if len(all_mean) > hist_len else []
            
            # 智能推断默认值
            # 如果缺少 next_check_days，尝试从预测窗口长度推断
            next_check_days = data.get('next_check_days')
            if next_check_days is None or next_check_days == 60:  # 60 是旧的错误默认值
                future_dates = data.get('future_dates', [])
                if len(future_dates) >= 2:
                    # 根据预测窗口长度推断（预测窗口 = 采样间隔 * 50%）
                    predicted_window = len(future_dates)
                    estimated_interval = predicted_window / 0.5  # 反推采样间隔
                    next_check_days = int(estimated_interval)
                    logger.info(f"  🔧 {indicator}: 从预测窗口({predicted_window}天)推断复查间隔 ≈ {next_check_days}天")
                else:
                    next_check_days = 30  # 保守的回退值
                    logger.warning(f"  ⚠️ {indicator}: 无法推断复查间隔，使用保守值 {next_check_days}天")
            
            mogp_results[indicator] = {
                'historical_dates': [pd.to_datetime(d) for d in data.get('historical_dates', [])],
                'historical_values': data.get('historical_values', []),
                
                'future_dates': [pd.to_datetime(d) for d in data.get('future_dates', [])],
                'predicted_mean': predicted_mean,
                'confidence_lower': data.get('confidence_lower', []),
                'confidence_upper': data.get('confidence_upper', []),
                
                'all_dates': [pd.to_datetime(d) for d in data.get('all_dates', [])],
                'all_predicted_mean': data.get('all_predicted_mean', []),
                'all_uncertainty': data.get('all_uncertainty', []),
                'all_trend': data.get('all_trend', []),

                'next_check_days': next_check_days,  # 🔧 使用修正后的值
                'confidence': data.get('confidence', 'medium'),
                'warning': data.get('warning')
            }
        
        logger.info(f"✅ MOGP结果已加载 (病人ID: {patient_id}, 更新时间: {last_updated})")
        return mogp_results, target_indicators, last_updated
        
    except Exception as e:
        logger.error(f"❌ 加载MOGP结果失败: PatientID={patient_id}: {e}", exc_info=True)
        return None, None, None
    finally:
        if conn:
            conn.close()

def label_lab_report(report_uuid: str, label: str):
    """
    为指定报告的所有条目打上用户反馈标签
    :param report_uuid: 报告的唯一ID
    :param label: 'benign', 'significant', 'lab_error' 之一
    """
    if label not in ['benign', 'significant', 'lab_error']:
        logger.error(f"无效的标签: {label}")
        raise ValueError("无效的标签")

    logger.info(f"准备为报告 {report_uuid} 打标签: '{label}'")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE lab_reports SET user_label = ? WHERE report_uuid = ?",
                (label, report_uuid)
            )
            conn.commit()

            # 立即验证写入是否成功
            c.execute(
                "SELECT user_label FROM lab_reports WHERE report_uuid = ? LIMIT 1",
                (report_uuid,)
            )
            result = c.fetchone()
            if result is None or result[0] != label:
                raise RuntimeError(f"标签写入验证失败：期望 {label}，实际 {result}")
            
            logger.info(f"✅ 报告 {report_uuid} 已标记为 '{label}'（已验证）。受影响行数: {c.rowcount}")
            
    except sqlite3.Error as e:
        logger.error(f"为 {report_uuid} 打标签失败: {e}", exc_info=True)
        conn.rollback()
        raise # 保留 raise，让上层知道出错了


def get_all_labels(patient_id: int) -> dict:
    """
    获取一个病人所有报告的UUID -> 标签 映射字典
    :return: dict (e.g., {'uuid1': 'benign', 'uuid2': 'significant'})
    """
    logger.debug(f"正在获取 PatientID={patient_id} 的所有用户标签...")

    try:
        with get_db_connection() as conn:

            query = """
                SELECT report_uuid, label
                FROM feedback_with_shap
                WHERE patient_id = ?
                  AND (report_uuid, timestamp) IN (
                      SELECT report_uuid, MAX(timestamp)
                      FROM feedback_with_shap
                      WHERE patient_id = ?
                      GROUP BY report_uuid
                  )
            """
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(patient_id, patient_id), 
                index_col='report_uuid'
            )

            if df.empty:
                logger.debug("未找到该病人的标签。")
                return {}

            labels_dict = df['label'].to_dict()
            logger.debug(
                f"成功获取 {len(labels_dict)} 个标签（来自 feedback_with_shap 表）。"
            )
            return labels_dict
    
    except sqlite3.DatabaseError as e:
        logger.warning(f"获取标签失败: {e}", exc_info=True)
        return {}
    

def save_similarity_feedback(current_uuid: str, indicator: str, is_similar: bool):
    """
    保存用户对历史相似度的反馈
    
    :param current_uuid: 当前报告的UUID
    :param indicator: 指标名称
    :param is_similar: 用户是否确认相似（True=相似，False=不相似）
    """
    logger.info(f"保存相似度反馈: UUID={current_uuid}, Indicator={indicator}, Similar={is_similar}")
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            # 插入或更新反馈
            c.execute('''
                INSERT OR REPLACE INTO similarity_feedback 
                (current_uuid, indicator, is_similar, feedback_date)
                VALUES (?, ?, ?, ?)
            ''', (current_uuid, indicator, 1 if is_similar else 0, datetime.now().isoformat()))
            
            conn.commit()
            logger.info(f"✅ 相似度反馈已保存")
    
    except sqlite3.Error as e:
        logger.error(f"保存相似度反馈失败: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise


def get_similarity_feedback_history(patient_id: int, indicator: str) -> dict:
    """
    获取某个指标的所有相似度反馈历史
    
    :param patient_id: 病人ID
    :param indicator: 指标名称
    :return: {current_uuid: is_similar} 的字典
    """
    logger.debug(f"获取相似度反馈历史: PatientID={patient_id}, Indicator={indicator}")
    
    try:
        with get_db_connection() as conn:
            # 联合查询，找出属于该病人的所有反馈
            query = """
                SELECT sf.current_uuid, sf.is_similar
                FROM similarity_feedback sf
                JOIN lab_reports lr ON sf.current_uuid = lr.report_uuid
                WHERE lr.patient_id = ? AND sf.indicator = ?
            """
            df = pd.read_sql_query(query, conn, params=(patient_id, indicator))
        
        if df.empty:
            return {}
        
        # 转换为字典
        result = {row['current_uuid']: bool(row['is_similar']) for _, row in df.iterrows()}
        logger.debug(f"找到 {len(result)} 条相似度反馈记录")
        return result
    
    except sqlite3.DatabaseError as e:
        logger.warning(f"获取相似度反馈历史失败: {e}", exc_info=True)
        return {}
    
def get_historical_labels_for_uuids(patient_id: int, uuids: list) -> dict:
    """
    【新增 V6.0】批量获取多个 UUID 的标签状态
    
    :param patient_id: 患者ID（用于安全验证）
    :param uuids: UUID 列表
    :return: {uuid: label} 字典
    """
    if not uuids:
        return {}
    
    logger.debug(f"批量查询 {len(uuids)} 个 UUID 的标签状态...")
    
    try:
        with get_db_connection() as conn:
            # 构建 IN 查询的占位符
            placeholders = ','.join('?' * len(uuids))
            query = f"""
                SELECT report_uuid, user_label
                FROM lab_reports
                WHERE patient_id = ? AND report_uuid IN ({placeholders})
                GROUP BY report_uuid
            """
            
            # 参数：patient_id + uuids
            params = [patient_id] + list(uuids)
            
            df = pd.read_sql_query(query, conn, params=params, index_col='report_uuid')
        
        if df.empty:
            return {}
        
        labels_dict = df['user_label'].to_dict()
        logger.debug(f"成功查询到 {len(labels_dict)} 个标签")
        return labels_dict
        
    except Exception as e:
        logger.error(f"批量查询标签失败: {e}", exc_info=True)
        return {}



# ==============================================================================
# 模型校准参数管理
# ==============================================================================

def save_calibrated_params(
    patient_id: int,
    model_name: str,
    marker: str,
    calibration_result: dict
):
    """
    保存模型校准参数到文件
    
    文件格式：patient_{id}_calibration_{model}_{marker}.json
    
    :param patient_id: 患者ID
    :param model_name: 模型名称（如"Lotka-Volterra 竞争模型"）
    :param marker: 标志物名称（如"CEA"）
    :param calibration_result: calibrate_model_with_scipy 返回的结果字典
    """
    logger.info(f"保存校准参数: PID={patient_id}, Model={model_name}, Marker={marker}")
    
    # 构建文件名（将空格替换为下划线）
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    calibration_file = os.path.join(
        config.MODELS_DIR,
        f"patient_{patient_id}_calibration_{safe_model_name}_{marker}.json"
    )
    
    try:
        # 准备保存数据
        save_data = {
            'model_name': model_name,
            'marker': marker,
            'calibrated_params': list(calibration_result['calibrated_params']),  # tuple转list
            'validation_error': calibration_result['validation_error'],
            'validation_date': calibration_result['validation_date'],
            'predicted_value': calibration_result['predicted_value'],
            'actual_value': calibration_result['actual_value'],
            'timestamp': datetime.now().isoformat(),
            'is_reliable': calibration_result['is_reliable'],
            'optimization_info': calibration_result.get('optimization_info', {})
        }
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 校准参数已保存至: {calibration_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 保存校准参数失败: {e}", exc_info=True)
        return False


def load_calibrated_params(patient_id: int, model_name: str, marker: str) -> dict:
    """
    【修复版】加载模型校准参数（从 JSON 文件）

    :param patient_id: 患者ID
    :param model_name: 模型名称
    :param marker: 标志物名称
    :return: {
        'params': tuple,           # 校准参数
        'error': float,            # 验证误差
        'predicted': float,        # 预测值
        'actual': float,           # 实际值
        'date': str,               # 验证日期
        'timestamp': str,          # 保存时间
        'is_reliable': bool        # 是否可靠
    } 或 None（如果文件不存在）
    """

    # 1. 构建文件路径（与 save_calibrated_params 保持一致）
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    calibration_file = os.path.join(
        config.MODELS_DIR,
        f"patient_{patient_id}_calibration_{safe_model_name}_{marker}.json"
    )
    
    # 2. 检查文件是否存在
    if not os.path.exists(calibration_file):
        logger.info(f"未找到校准参数文件：{calibration_file}")
        return None
    
    try:
        # 3. 读取 JSON 文件
        with open(calibration_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 4. 提取参数（从列表转换为元组）
        params_tuple = tuple(data['calibrated_params'])
        
        # 5. 【版本兼容性检查】确保参数数量匹配
        model_config = tumor_models.MODEL_FACTORY.get(model_name)
        if not model_config:
            logger.error(f"未知模型: {model_name}")
            return None
        
        expected_n_params = len(model_config['params'])
        
        if len(params_tuple) != expected_n_params:
            logger.warning(
                f"⚠️ 校准参数版本不兼容！\n"
                f"  - 文件中参数数量: {len(params_tuple)}\n"
                f"  - 当前模型需要: {expected_n_params}\n"
                f"  - 保存时间: {data.get('timestamp', '未知')}\n"
                f"  → 忽略旧参数，返回 None 触发重新校准"
            )
            return None
        
        # 6. 返回格式化结果
        logger.info(
            f"✅ 加载校准参数: {model_name} - {marker}\n"
            f"  - 验证误差: {data['validation_error']:.1%}\n"
            f"  - 验证日期: {data['validation_date']}\n"
            f"  - 保存时间: {data['timestamp']}"
        )
        
        return {
            'params': params_tuple,
            'error': data['validation_error'],
            'predicted': data['predicted_value'],
            'actual': data['actual_value'],
            'date': data['validation_date'],
            'timestamp': data['timestamp'],
            'is_reliable': data.get('is_reliable', False)
        }
        
    except FileNotFoundError:
        logger.info(f"校准参数文件不存在: {calibration_file}")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {calibration_file}, 错误: {e}")
        return None
        
    except KeyError as e:
        logger.error(f"校准参数文件格式不完整，缺少字段: {e}")
        return None
        
    except Exception as e:
        logger.error(f"加载校准参数失败: {e}", exc_info=True)
        return None


def delete_calibrated_params(patient_id: int, model_name: str, marker: str) -> bool:
    """
    删除校准参数文件（用于重新校准或清理）
    
    :return: True（成功删除或文件不存在），False（删除失败）
    """
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    calibration_file = os.path.join(
        config.MODELS_DIR,
        f"patient_{patient_id}_calibration_{safe_model_name}_{marker}.json"
    )
    
    if not os.path.exists(calibration_file):
        logger.debug(f"校准参数文件不存在，无需删除: {calibration_file}")
        return True
    
    try:
        os.remove(calibration_file)
        logger.info(f"✅ 已删除校准参数文件: {calibration_file}")
        return True
    except Exception as e:
        logger.error(f"❌ 删除校准参数文件失败: {e}", exc_info=True)
        return False
    

# 用户反馈与 SHAP 归因管理
def save_feedback_with_shap(
    patient_id: int,
    report_uuid: str,
    indicator: str,
    label: str,
    shap_values_dict: dict,
    observation_uuid: str = None,
    pattern_type: str = None,
    shap_type: str = 'proxy'
):
    """
    【统一反馈系统】保存用户反馈及其对应的 SHAP 归因值（带归因类型标记）
    
    :param patient_id: 患者ID
    :param report_uuid: 报告UUID
    :param indicator: 指标名称（如 "癌胚抗原 CEA"）
    :param label: 用户标签 ('benign', 'significant', 'lab_error')
    :param shap_values_dict: SHAP 归因字典 {feature: shap_value}
    :param observation_uuid: 观察项UUID（可选，用于去重）
    :param pattern_type: 警报类型（如 "z_score", "model_anomaly", "heuristic"）
    :param shap_type: 'real' = 来自模型的真实 SHAP
                      'proxy' = 代理归因（100% 单指标）    
    :return: 保存的 UUID
    """
    logger.info(
        f"保存反馈 (PID={patient_id}, UUID={report_uuid[:8]}..., "
        f"Label={label}, SHAP Type={shap_type})"
    )
    
    # 验证标签
    if label not in ['benign', 'significant', 'lab_error']:
        raise ValueError(f"无效的标签: {label}")
    
    # 验证 shap_type
    if shap_type not in ['real', 'proxy']:
        logger.warning(f"无效的 shap_type: {shap_type}，回退到 'proxy'")
        shap_type = 'proxy'
    
    # 自动校正：如果标记为'real'但只有单特征，自动改为'proxy'
    if shap_type == 'real' and len(shap_values_dict) == 1:
        logger.warning(
            f"⚠️ 标记为'real'但只有单特征归因 ({list(shap_values_dict.keys())}), "
            f"自动修正为'proxy'"
        )
        shap_type = 'proxy'
    
    # 生成或使用提供的 UUID
    if not observation_uuid:
        observation_uuid = str(uuid.uuid4())
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # 序列化 SHAP 值
            shap_json = json.dumps(shap_values_dict, ensure_ascii=False)
            
            # 插入或替换
            c.execute('''
                INSERT OR REPLACE INTO feedback_with_shap 
                (observation_uuid, patient_id, report_uuid, indicator, label, 
                 shap_values, pattern_type, shap_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                observation_uuid, 
                patient_id, 
                report_uuid, 
                indicator,
                label, 
                shap_json,
                pattern_type,
                shap_type
            ))
            
            conn.commit()
            logger.info(
                f"✅ 反馈已保存 (UUID={observation_uuid[:8]}..., "
                f"SHAP Type={shap_type})"
            )
            return observation_uuid
            
    except Exception as e:
        logger.error(f"保存反馈失败: {e}", exc_info=True)
        raise


def load_all_feedback_with_shap(patient_id: int) -> list:
    """
    加载患者所有历史反馈及其 SHAP 归因值（含归因类型）
    
    返回格式:
    [
        {
            'observation_uuid': 'xxx',
            'report_uuid': 'yyy',
            'indicator': '癌胚抗原 CEA',
            'label': 'significant',
            'timestamp': datetime(2024, 1, 1),
            'shap_values': {'癌胚抗原 CEA': 0.8, 'WBC': 0.2},
            'pattern_type': 'baseline_deviation',
            'shap_type': 'proxy'
        },
        ...
    ]
    """
    logger.debug(f"加载患者反馈 (PID={patient_id})...")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT observation_uuid, report_uuid, indicator, label, 
                       timestamp, shap_values, pattern_type, shap_type
                FROM feedback_with_shap
                WHERE patient_id = ? 
                  AND label IN ('significant', 'benign', 'lab_error')
                ORDER BY timestamp DESC
            '''
            
            cursor.execute(query, (patient_id,))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                obs_uuid, report_uuid, indicator, label, timestamp_str, shap_json, pattern_type, shap_type = row
                
                try:
                    shap_dict = json.loads(shap_json)
                except json.JSONDecodeError:
                    logger.warning(f"反馈 {obs_uuid} 的 SHAP 值解析失败，跳过")
                    continue
                
                results.append({
                    'observation_uuid': obs_uuid,
                    'report_uuid': report_uuid,
                    'indicator': indicator,
                    'label': label,
                    'timestamp': pd.to_datetime(timestamp_str),
                    'shap_values': shap_dict,
                    'pattern_type': pattern_type,
                    'shap_type': shap_type
                })
            
            logger.info(f"✅ 加载了 {len(results)} 条反馈 (PID={patient_id})")
            return results
            
    except Exception as e:
        logger.error(f"加载反馈失败: {e}", exc_info=True)
        return []


def get_feedback_summary(patient_id: int) -> dict:
    """
    【统一反馈系统】获取患者反馈的统计摘要
    
    :return: {
        'total': 总反馈数,
        'benign': 良性波动数,
        'significant': 重要变化数,
        'lab_error': 数据错误数,
        'latest_timestamp': 最新反馈时间,
        'by_pattern': {'z_score': 5, 'model_anomaly': 2, ...}
    }
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 统计各类标签数量
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN label = 'benign' THEN 1 ELSE 0 END) as benign,
                    SUM(CASE WHEN label = 'significant' THEN 1 ELSE 0 END) as significant,
                    SUM(CASE WHEN label = 'lab_error' THEN 1 ELSE 0 END) as lab_error,
                    MAX(timestamp) as latest_timestamp
                FROM feedback_with_shap
                WHERE patient_id = ?
            ''', (patient_id,))
            
            row = cursor.fetchone()
            
            # 按警报类型统计
            cursor.execute('''
                SELECT pattern_type, COUNT(*) as count
                FROM feedback_with_shap
                WHERE patient_id = ?
                GROUP BY pattern_type
            ''', (patient_id,))
            
            by_pattern = {r[0]: r[1] for r in cursor.fetchall()}
            
            if row and row[0] > 0:
                return {
                    'total': row[0],
                    'benign': row[1] or 0,
                    'significant': row[2] or 0,
                    'lab_error': row[3] or 0,
                    'latest_timestamp': pd.to_datetime(row[4]) if row[4] else None,
                    'by_pattern': by_pattern
                }
            else:
                return {
                    'total': 0,
                    'benign': 0,
                    'significant': 0,
                    'lab_error': 0,
                    'latest_timestamp': None,
                    'by_pattern': {}
                }
                
    except Exception as e:
        logger.error(f"获取反馈摘要失败: {e}", exc_info=True)
        return {
            'total': 0, 'benign': 0, 'significant': 0, 'lab_error': 0,
            'latest_timestamp': None, 'by_pattern': {}
        }
