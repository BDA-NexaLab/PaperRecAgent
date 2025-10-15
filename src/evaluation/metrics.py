# src/evaluation/metrics.py
import numpy as np
import pandas as pd
import ast

# -------------------------
# 유틸: 어떤 형태든 list로 변환
# -------------------------
def _as_list(cell):
    """셀 값이 list/Series/str/단일값 어떤 형태여도 list로 변환."""
    if isinstance(cell, list):
        return cell
    if isinstance(cell, (pd.Series, pd.Index)):
        return list(cell)
    if pd.isna(cell):
        return []
    # numpy 배열/리스트류
    if hasattr(cell, "tolist"):
        try:
            return cell.tolist()
        except Exception:
            pass
    if isinstance(cell, str):
        s = cell.strip()
        # JSON/파이썬 리스트 형태면 파싱 시도: ["a","b"] 또는 ['a','b']
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                pass
        # 콤마 구분 문자열 처리
        return [t.strip() for t in s.split(",") if t.strip()]
    # 단일 값
    return [cell]

# -------------------------
# 유틸: GT 컬럼 감지
# -------------------------
_GT_ID_CANDIDATES = ["입력id", "input_id", "입력ID", "query_id", "id"]
_GT_RESULT_CANDIDATES = ["추천결과", "결과", "정답", "gt", "answer", "label", "target", "output"]

def _find_id_col(df: pd.DataFrame) -> str:
    for c in _GT_ID_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"ID 컬럼을 찾지 못했습니다. 후보={_GT_ID_CANDIDATES}, 실제={list(df.columns)}")

def _find_gt_col(df: pd.DataFrame) -> str:
    for c in _GT_RESULT_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"정답(추천결과) 컬럼을 찾지 못했습니다. 후보={_GT_RESULT_CANDIDATES}, 실제={list(df.columns)}")

# -------------------------
# nDCG@K
# -------------------------
def ndcg_at_k(id: str, recommended: pd.DataFrame, test_data: pd.DataFrame, k: int = 5) -> float:
    id = str(id)

    # GT 컬럼 결정
    id_col = _find_id_col(test_data)          # 네 파일에선 '입력id'
    gt_col = _find_gt_col(test_data)          # 네 파일에선 '추천결과'

    # 해당 id의 GT 수집
    test_rows = test_data[test_data[id_col].astype(str) == id]
    if test_rows.empty:
        return 0.0

    # 셀마다 다양한 형식을 평탄화해 집합으로
    true_items = []
    for cell in test_rows[gt_col]:
        true_items.extend(_as_list(cell))
    true_set = set(str(x) for x in true_items)

    # 추천 결과 셀 가져오기 (index 타입 차이 방어)
    try:
        cell = recommended.loc[id, "결과"]
    except KeyError:
        # 인덱스를 문자열로 바꿔 재조회
        try:
            cell = recommended.set_index(recommended.index.astype(str)).loc[id, "결과"]
        except KeyError:
            # 혹시 추천 DataFrame이 '추천결과' 컬럼을 쓴 경우도 지원
            try:
                cell = recommended.loc[id, "추천결과"]
            except Exception:
                return 0.0

    rec_list = [str(x) for x in _as_list(cell)[:k]]

    # DCG
    dcg = 0.0
    for i, item in enumerate(rec_list):
        if item in true_set:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(true_set), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0

# -------------------------
# Recall@K
# -------------------------
def recall_at_k(id: str, recommended: pd.DataFrame, test_data: pd.DataFrame, k: int = 5) -> float:
    id = str(id)

    # GT 컬럼 결정
    id_col = _find_id_col(test_data)          # '입력id'
    gt_col = _find_gt_col(test_data)          # '추천결과'

    test_rows = test_data[test_data[id_col].astype(str) == id]
    if test_rows.empty:
        return 0.0

    true_items = []
    for cell in test_rows[gt_col]:
        true_items.extend(_as_list(cell))
    true_set = set(str(x) for x in true_items)

    try:
        cell = recommended.loc[id, "결과"]
    except KeyError:
        try:
            cell = recommended.set_index(recommended.index.astype(str)).loc[id, "결과"]
        except KeyError:
            try:
                cell = recommended.loc[id, "추천결과"]
            except Exception:
                return 0.0

    rec_list = _as_list(cell)[:k]
    rec_set = set(str(x) for x in rec_list)

    hits = len(rec_set & true_set)
    return hits / len(true_set) if len(true_set) > 0 else 0.0
