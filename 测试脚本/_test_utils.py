import os
import sys
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@contextmanager
def isolated_config():
    import config
    import data_manager

    old_db = config.DB_FILE
    old_models = config.MODELS_DIR

    tmp_root = PROJECT_ROOT / "测试脚本" / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    tmp_path = tmp_root / f"caretrace_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    config.DB_FILE = str(tmp_path / "test_health_data.db")
    config.MODELS_DIR = str(tmp_path / "models")
    data_manager.init_db()
    try:
        yield tmp_path
    finally:
        config.DB_FILE = old_db
        config.MODELS_DIR = old_models
        shutil.rmtree(tmp_path, ignore_errors=True)


def make_items(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"指标名称": name, "检测值": value} for name, value in values.items()]
    )


def make_patient_frame(values, phase="稳定监控期", start="2026-01-01", freq="7D"):
    index = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.DataFrame(
        {
            "phase": [phase] * len(values),
            "report_uuid": [f"r{i}" for i in range(len(values))],
            "癌胚抗原 CEA": values,
            "白细胞计数": [5.0 + i * 0.05 for i in range(len(values))],
            "中性粒细胞绝对数": [3.0 + i * 0.02 for i in range(len(values))],
            "淋巴细胞绝对数": [1.5 + i * 0.01 for i in range(len(values))],
            "血小板计数": [220 + i for i in range(len(values))],
            "单核细胞绝对数": [0.4 + i * 0.005 for i in range(len(values))],
        },
        index=index,
    )
