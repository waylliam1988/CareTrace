# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# ocr_importer.py
"""
本地化验单图片 OCR 解析。

设计原则：
- OCR 只做“预填充”，不直接替代人工确认。
- 日期优先采用采样时间，其次报告时间，再其次申请时间。
- 指标名称只映射到 config.LAB_REPORT_CONFIG 中的标准名，避免把错误文本写成新指标。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

import config


@dataclass
class OCRTextLine:
    text: str
    confidence: float = 1.0
    y_center: float = 0.0
    x_left: float = 0.0


DATE_PRIORITY = ["采样时间", "报告时间", "申请时间"]


def rapidocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def run_ocr_on_image_bytes(image_bytes: bytes) -> list[OCRTextLine]:
    """用本地 RapidOCR 识别图片字节，返回按行合并后的文本。"""
    try:
        import cv2
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError("未检测到 cv2 或 rapidocr_onnxruntime，无法执行本地 OCR。") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片，请确认文件格式为 PNG/JPG/WebP。")

    result, _ = RapidOCR()(image)
    return _merge_ocr_boxes(result or [])


def _merge_ocr_boxes(result_rows: Iterable) -> list[OCRTextLine]:
    cells = []
    for row in result_rows:
        if len(row) < 2:
            continue
        box, text = row[0], str(row[1]).strip()
        confidence = float(row[2]) if len(row) >= 3 else 1.0
        if not text:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        cells.append(
            OCRTextLine(
                text=text,
                confidence=confidence,
                y_center=sum(ys) / len(ys),
                x_left=min(xs),
            )
        )

    if not cells:
        return []

    cells.sort(key=lambda item: (item.y_center, item.x_left))
    median_height = max(12.0, _estimate_line_height(cells))
    row_tolerance = median_height * 0.65
    grouped: list[list[OCRTextLine]] = []

    for cell in cells:
        if not grouped or abs(grouped[-1][0].y_center - cell.y_center) > row_tolerance:
            grouped.append([cell])
        else:
            grouped[-1].append(cell)

    lines = []
    for group in grouped:
        group.sort(key=lambda item: item.x_left)
        lines.append(
            OCRTextLine(
                text=" ".join(item.text for item in group),
                confidence=min(item.confidence for item in group),
                y_center=sum(item.y_center for item in group) / len(group),
                x_left=min(item.x_left for item in group),
            )
        )
    return lines


def _estimate_line_height(cells: list[OCRTextLine]) -> float:
    ys = sorted(item.y_center for item in cells)
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 3]
    if not diffs:
        return 24.0
    return float(pd.Series(diffs).median())


def parse_report_text_lines(lines: Iterable[str | OCRTextLine], source_name: str = "") -> dict:
    """解析 OCR 行文本，返回日期、匹配指标和未匹配文本。"""
    normalized_lines = [
        line if isinstance(line, OCRTextLine) else OCRTextLine(str(line))
        for line in lines
    ]
    dates = _extract_dates([line.text for line in normalized_lines])
    selected_date_source, selected_date = _select_report_date(dates)

    matched_items = []
    unmatched_lines = []
    seen = set()

    for line in normalized_lines:
        parsed = _parse_item_line(line.text)
        if not parsed:
            text = line.text.strip()
            if _looks_like_possible_result_row(text):
                unmatched_lines.append(text)
            continue

        item_name = parsed["indicator"]
        if item_name in seen:
            continue
        seen.add(item_name)

        matched_items.append(
            {
                "指标名称": item_name,
                "检测值": parsed["value"],
                "OCR文本": line.text,
                "识别置信度": line.confidence,
                "匹配方式": parsed["match_text"],
            }
        )

    matched_df = pd.DataFrame(matched_items)
    if not matched_df.empty:
        matched_df = matched_df.sort_values(by=["指标名称"]).reset_index(drop=True)

    return {
        "source_name": source_name,
        "report_date": selected_date,
        "date_source": selected_date_source,
        "dates": dates,
        "items": matched_df,
        "unmatched_lines": unmatched_lines,
        "raw_text": "\n".join(line.text for line in normalized_lines),
    }


def _extract_dates(lines: list[str]) -> dict[str, str]:
    text = "\n".join(lines)
    normalized = (
        text.replace("：", ":")
        .replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .replace(".", "-")
    )

    date_map = {}
    pattern = re.compile(
        r"(申请时间|采样时间|报告时间)\s*[:：]?\s*"
        r"(\d{4}-\d{1,2}-\d{1,2})\s*"
        r"([0-2]?\d:\d{2}(?::\d{2})?)?"
    )
    for label, date_part, time_part in pattern.findall(normalized):
        date_string = date_part
        if time_part:
            date_string = f"{date_part} {time_part}"
        parsed = _parse_datetime(date_string)
        if parsed:
            date_map[label] = parsed.isoformat(sep=" ")
    return date_map


def _parse_datetime(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _select_report_date(dates: dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    for key in DATE_PRIORITY:
        if key in dates:
            return key, dates[key][:10]
    return None, None


def _parse_item_line(text: str) -> Optional[dict]:
    compact = _normalize_indicator_text(text)
    if not compact or any(token in compact for token in ["项目结果参考区间单位", "报告详情", "查看报告"]):
        return None
    if "+" in compact and not _has_result_context(compact):
        return None

    inferred = _parse_special_context_line(text, compact)
    if inferred:
        return inferred

    match = _find_indicator_match(compact)
    if not match:
        return None

    item_name, match_text, start, end = match
    after = compact[end:]
    value = _extract_first_numeric_token(text)
    if value is None:
        value = _extract_first_number(after)
    if value is None:
        # 有些 OCR 会只匹配短别名，值仍在别名之后较远位置。
        value = _extract_first_number(compact[start + len(match_text):])
    if value is None:
        return None

    return {"indicator": item_name, "value": value, "match_text": match_text}


def _has_result_context(compact: str) -> bool:
    context_tokens = ["~", "<", ">", "<=", ">=", "IU/L", "U/ML", "NG/ML", "PG/ML", "PMOL/L", "MMOL/L", "10^", "G/L", "%"]
    return any(token in compact for token in context_tokens)


def _parse_special_context_line(text: str, compact: str) -> Optional[dict]:
    """处理 OCR 丢失项目名但保留参考区间的少数固定格式行。"""
    # 该院生化截图中钠[NA]偶尔会被识别成 []，但参考区间和单位仍能保留。
    if (str(text).strip().startswith("[]") or compact.startswith("134")) and "137" in compact and "147" in compact and "MMOL/L" in compact:
        value = _extract_first_numeric_token(text)
        if value is not None:
            return {"indicator": "钠 NA", "value": value, "match_text": "钠参考区间推断"}
    return None


@lru_cache(maxsize=1)
def _indicator_candidates() -> tuple[tuple[str, str], ...]:
    candidates = []
    for items in config.LAB_REPORT_CONFIG.values():
        for item in items:
            names = [item["name"], *item.get("aliases", [])]
            for name in names:
                norm = _normalize_indicator_text(name)
                if _valid_alias(norm):
                    candidates.append((norm, item["name"]))

            chinese_only = _normalize_indicator_text(re.sub(r"[A-Za-z0-9µμ/().+\-]+", "", item["name"]))
            if _valid_alias(chinese_only):
                candidates.append((chinese_only, item["name"]))

            percent_alias = _normalize_indicator_text(item["name"].replace("百分比", "%"))
            if percent_alias != _normalize_indicator_text(item["name"]) and _valid_alias(percent_alias):
                candidates.append((percent_alias, item["name"]))

    dedup = {}
    for alias, canonical in candidates:
        dedup.setdefault(alias, canonical)
    return tuple(sorted(dedup.items(), key=lambda pair: len(pair[0]), reverse=True))


@lru_cache(maxsize=1)
def _indicator_alias_set() -> frozenset[str]:
    return frozenset(alias for alias, _ in _indicator_candidates())


def _valid_alias(alias: str) -> bool:
    if not alias:
        return False
    if len(alias) >= 2:
        return True
    return bool(re.search(r"[\u4e00-\u9fff]", alias))


def _find_indicator_match(compact_line: str) -> Optional[tuple[str, str, int, int]]:
    if "红细胞体积分布宽度" in compact_line:
        if "CV" in compact_line:
            position = compact_line.find("红细胞体积分布宽度")
            return "红细胞体积分布宽度-CV", "红细胞体积分布宽度CV", position, position + len("红细胞体积分布宽度")
        if "SD" in compact_line:
            position = compact_line.find("红细胞体积分布宽度")
            return "红细胞体积分布宽度-SD", "红细胞体积分布宽度SD", position, position + len("红细胞体积分布宽度")

    for alias, canonical in _indicator_candidates():
        position = compact_line.find(alias)
        if position >= 0:
            return canonical, alias, position, position + len(alias)
    return None


def _extract_first_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").replace("O", "0").replace("o", "0")
    match = re.search(r"(?<![A-Z])[-+]?\d+(?:\.\d+)?(?![A-Z])", cleaned)
    if not match:
        return None
    value_text = match.group(0)
    if "." in value_text and match.end() < len(cleaned) and cleaned[match.end()] == ".":
        integer_part, decimal_part = value_text.split(".", 1)
        if len(decimal_part) > 1:
            value_text = f"{integer_part}.{decimal_part[:-1]}"
    try:
        return float(value_text)
    except ValueError:
        return None


def _extract_first_numeric_token(text: str) -> Optional[float]:
    for token in re.split(r"\s+", str(text).strip()):
        if not token or re.search(r"[\u4e00-\u9fff]", token):
            if _normalize_indicator_text(token) in _indicator_alias_set():
                continue
            trailing_value = re.search(r"(\d+(?:\.\d+)?)\s*$", token)
            previous_char = token[trailing_value.start() - 1] if trailing_value and trailing_value.start() > 0 else ""
            if trailing_value and re.search(r"[\u4e00-\u9fff]", previous_char):
                try:
                    return float(trailing_value.group(1))
                except ValueError:
                    pass
            continue
        if re.search(r"[A-Za-z]", token):
            continue
        token = (
            token.replace("↑", "")
            .replace("↓", "")
            .replace("<=", "")
            .replace(">=", "")
            .replace("<", "")
            .replace(">", "")
            .replace("≤", "")
            .replace("≥", "")
            .replace(",", "")
        )
        match = re.search(r"[-+]?\d+(?:\.\d+)?", token)
        if not match:
            continue
        try:
            return float(match.group(0))
        except ValueError:
            continue
    return None


def _normalize_indicator_text(text: str) -> str:
    normalized = str(text).upper()
    replacements = {
        "×": "*",
        "－": "-",
        "—": "-",
        "–": "-",
        "（": "(",
        "）": ")",
        "【": "",
        "】": "",
        "[": "",
        "]": "",
        "★": "",
        "☆": "",
        "↑": "|",
        "↓": "|",
        "≤": "<=",
        "≥": ">=",
        "μ": "U",
        "µ": "U",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = normalized.replace("C-反应蛋白", "C反应蛋白")
    return re.sub(r"\s+", "", normalized)


def _looks_like_possible_result_row(text: str) -> bool:
    if not re.search(r"\d", text):
        return False
    blacklist = ["申请时间", "采样时间", "报告时间", "项目", "参考区间", "单位"]
    return not any(token in text for token in blacklist)
