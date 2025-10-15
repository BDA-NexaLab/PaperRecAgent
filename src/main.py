# src/main.py
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from dotenv import load_dotenv

# ---- 프로젝트 내부 모듈 ----
from utils.text_cleaner import clean_text
from recommendation.recommender_co_eu import recommend_coeu_with_llm
from utils.weights_search import grid_search_weights
from utils.final_report import make_final_report
from data_collection.fetch_data import collect_data
from evaluation.metrics import ndcg_at_k, recall_at_k
from llm_interface.query_generator import generate_query
from utils.text_cleaner import prepare_query

# =========================
# 설정/경로 (루트 자동계산)
# =========================
ROOT = Path(__file__).resolve().parents[1]     # .../BDA
DATA_DIR = ROOT / "data"

TEST_ID_CSV   = DATA_DIR / "test_input_id.csv"
TEST_GT_CSV   = DATA_DIR / "final_test_data.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


def main():
    # 0) env 로드 + LLM URL/Headers 준비
    load_dotenv()

    url = os.getenv("URL")
    headers_raw = os.getenv("HEADERS")
    headers = None
    if headers_raw:
        try:
            headers = json.loads(headers_raw)  # 문자열 -> dict
        except json.JSONDecodeError:
            print("[WARN] .env의 HEADERS JSON 파싱 실패. headers=None 로 진행")

    # 1) 테스트 ID 로드
    if not TEST_ID_CSV.exists():
        raise FileNotFoundError(f"테스트 ID 파일을 찾을 수 없습니다: {TEST_ID_CSV}")
    test_id_df = pd.read_csv(TEST_ID_CSV)

    # 예: 두 번째 행의 id를 문자열로 추출 (열 이름 모르면 squeeze로 임시)
    id_val = str(test_id_df.iloc[1].squeeze())

    # 2) 모델/형태소기
    embed_model = SentenceTransformer(MODEL_NAME)
    okt = Okt()

    # 3) 레퍼런스 수집
    reference = collect_data(id_val, "DataOn", row_count=1, return_type="dict")

    # 4) LLM으로 쿼리 생성 (재시도 + 폴백)
    MAX_Q_RETRY = 5
    ko_query, en_query = [], []

    for attempt in range(MAX_Q_RETRY):
        try:
            ko_query, en_query = generate_query(reference, url=url, headers=headers)
        except Exception as e:
            print(f"[WARN] generate_query 실패: {e}")
            ko_query, en_query = [], []

        if ko_query and en_query:
            break
        print(f"[INFO] 쿼리 재시도 {attempt+1}/{MAX_Q_RETRY}")

    if not (ko_query and en_query):
        # 간단 폴백: reference의 텍스트 일부를 정제하여 기본 쿼리 구성
        ref_text = str(reference)[:300]
        default_q = clean_text(ref_text) or "인공지능 연구"
        ko_query = [default_q]
        en_query = ["artificial intelligence"]

    # 쿼리 정제
    ko_query_limit = [prepare_query(q) for q in ko_query]
    en_query_limit = [prepare_query(q) for q in en_query]

    # 5) 후보 수집 (재시도 한도, 무한루프 방지)
    def try_collect(q_ko: str, q_en: str, row_count=50):
        dko = collect_data(q_ko, "DataOn",    row_count=row_count, return_type="dict") or []
        den = collect_data(q_en, "DataOn",    row_count=row_count, return_type="dict") or []
        sko = collect_data(q_ko, "ScienceOn", row_count=row_count, return_type="dict") or []
        sen = collect_data(q_en, "ScienceOn", row_count=row_count, return_type="dict") or []
        return dko + den + sko + sen

    MAX_C_RETRY = 3
    candidates = []
    for attempt in range(MAX_C_RETRY):
        candidates = try_collect(ko_query_limit[0], en_query_limit[0], row_count=50)
        if candidates:
            print(f"[OK] candidates 수집 완료: {len(candidates)}개")
            break
        print(f"[INFO] candidates=0. 재시도 {attempt+1}/{MAX_C_RETRY}")

    if not candidates:
        print("[ERROR] 후보를 수집하지 못했습니다. URL/KEY/쿼리를 점검하세요.")
        return  # 종료 (무한루프 방지)

    # 6) 가중치 그리드서치 
    best_weights = grid_search_weights(
        reference, candidates, embed_model,
        okt=okt, url=url, headers=headers,
        num_topics=5, top_k=TOP_K, use_llm=False
    )
    print(">> BEST WEIGHTS =", best_weights)

    # 7) 최종 추천 (LLM 이유 생성 ON)
    final_results, topics = recommend_coeu_with_llm(
        reference, candidates, embed_model, okt=okt, url=url, headers=headers,
        num_topics=5, top_k=TOP_K, weights=best_weights, use_llm=True
    )

    print("=== FINAL (Top-K) ===")
    for r in final_results:
        print(r)

    print("=== TOPIC SUMMARY ===")
    for t in topics:
        print("-", t)

    # 8) 최종 리포트
    report_df = make_final_report(final_results, candidates, desc_maxlen=300)
    print(report_df.to_string(index=False))

    # 9) 정량 평가
    if not TEST_GT_CSV.exists():
        print(f"[WARN] GT 파일 없음: {TEST_GT_CSV} — 정량 평가를 건너뜀")
        return

    test_data = pd.read_csv(TEST_GT_CSV)

    # metrics 기대 형식에 맞게 어댑트: index=id, '결과' = 추천 ID 리스트
    rec_ids = [str(item.get("id") or item.get("svc_id") or item.get("doc_id")) for item in final_results]
    recommended = pd.DataFrame({"결과": [rec_ids]}, index=[id_val])

    print("=== 정량 평가 결과 ===")
    print("NDCG@5:", ndcg_at_k(id_val, recommended, test_data, k=5))
    print("Recall@5:", recall_at_k(id_val, recommended, test_data, k=5))


if __name__ == "__main__":
    main()
