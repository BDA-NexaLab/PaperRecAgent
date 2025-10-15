import numpy as np
from itertools import product
from recommendation.recommender_co_eu import recommend_coeu_with_llm

# 가중치 그리드서치 함수
def grid_search_weights(reference_docs, candidate_docs, embed_model, okt=None,url=None, headers=None,
                        num_topics=5, top_k=5, use_llm=False,
                        weight_steps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """
    여러 (w_t, w_c, w_e) 조합을 돌려 평균 점수가 가장 높은 가중치를 리턴하는 함수
    """
    best_score = -1
    best_weights = (0.2, 0.4, 0.4)  # 초기 기본값

    for w_t, w_c, w_e in product(weight_steps, repeat=3):
        if abs((w_t + w_c + w_e) - 1.0) > 1e-6:
            continue

        results, _ = recommend_coeu_with_llm(
            reference_docs, candidate_docs, embed_model,
            okt=okt,url=url, headers=headers, num_topics=num_topics, top_k=top_k,
            weights=(w_t, w_c, w_e), use_llm=use_llm
        )

        if not results:
            continue

        avg_score = float(np.mean([r["score"] for r in results]))
        if avg_score > best_score:
            best_score = avg_score
            best_weights = (w_t, w_c, w_e)

    print(f"[GridSearch] 최적 가중치={best_weights}, 평균 점수={round(best_score,3)}")
    return best_weights  # 가중치 3개만 반환