import pandas as pd

# === 최종 리포트 표 생성 ===
def make_final_report(final_results, candidate_docs, desc_maxlen=300):
    # 후보 메타데이터(설명/URL) 조회용 인덱스
    by_title = {}
    for c in candidate_docs:
        t = (c.get('title','') or '').strip()
        if t:  # 중복 타이틀이 있을 수 있어 첫 항목 우선
            by_title.setdefault(t, c)

    rows = []
    for r in final_results:
        title = (r.get('title','') or '').strip()
        meta  = by_title.get(title, {})
        desc  = (meta.get('description','') or '').strip()
        url   = (meta.get('url') or meta.get('link') or meta.get('pdf_url') or meta.get('doi') or '').strip()
        reason = r.get('reason','')
        if 'LLM' in reason and 'skipped' in reason.lower():
            reason = ''

        score = float(r.get('score', 0.0))
        if score >= 0.8:
            recommendation_level = "강추"
        elif score >= 0.5:
            recommendation_level = "추천"
        elif score >= 0.3:
            recommendation_level = "참고"
        else:
            recommendation_level = "관련없음"

        rows.append({
            "구분": r.get('division') or meta.get('division',''),
            "제목": title,
            #"설명": desc[:desc_maxlen],
            "점수": score,
            "추천수준": recommendation_level,
            "추천사유": reason,
            "URL": url
        })

    df = pd.DataFrame(rows)
    # 점수 내림차순 정렬
    if not df.empty:
        df = df.sort_values("점수", ascending=False, kind="mergesort").reset_index(drop=True)
    return df
