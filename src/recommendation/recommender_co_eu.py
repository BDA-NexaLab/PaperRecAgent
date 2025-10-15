import numpy as np
from utils.text_cleaner import clean_text
from utils.embedding_utils import get_embeddings
from utils.topic_utils import extract_topics, summarize_topics
from utils.similarity_utils import euclidean_similarity, combine_scores
from llm_interface.reason_generator import generate_reason_llm
from utils.similarity_utils import cosine_similarity_score


#정규화 함수
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

# 토픽+ 코사인 + 유클리디안
def recommend_coeu_with_llm(reference_docs, candidate_docs, embed_model,
                            okt=None, url=None, headers=None,
                            num_topics=5, alpha=0.6, top_k=5,
                            weights=(0.2, 0.4, 0.4), use_llm=True):
    """
    Coeu 기반 추천 함수
    (Topic + Cosine + Euclidean 유사도 조합)
    """
    # --- 후보 필터링 ---
    candidate_docs = [c for c in candidate_docs
                      if c.get('title', '').strip() != reference_docs[0].get('title', '').strip()]
    if not candidate_docs:
        return [], []

    # --- 텍스트 정리 ---
    ref_texts = [clean_text(r.get('title', '') + " " + r.get('description', ''), okt)
                 for r in reference_docs]
    cand_texts = [clean_text(c.get('title', '') + " " + c.get('description', ''), okt)
                  for c in candidate_docs]

    # --- 임베딩 계산 ---
    all_texts = (ref_texts + cand_texts) * 10
    emb = get_embeddings(all_texts, embed_model)

    # --- 토픽 추출 ---
    topic_model, topics, topic_vectors = extract_topics(all_texts, emb, num_topics=num_topics)
    doc_topics = topics[:len(all_texts)//10]

    # --- 벡터 준비 ---
    ref_vec = np.mean([topic_vectors.get(t, np.zeros(emb.shape[1]))
                       for t in doc_topics[:len(ref_texts)]], axis=0)
    cand_vecs = np.array([topic_vectors.get(t, np.zeros(emb.shape[1]))
                          for t in doc_topics[len(ref_texts):]])

    ref_emb = emb[:len(ref_texts)]
    cand_emb = emb[len(ref_texts):len(ref_texts)+len(cand_texts)]

    # --- 유사도 계산 ---
    topic_sim   = cosine_similarity_score(cand_vecs, ref_vec.reshape(1, -1))
    text_co_sim = cosine_similarity_score(cand_emb, np.mean(ref_emb, axis=0).reshape(1, -1))
    text_eu_sim = euclidean_similarity(cand_emb, np.mean(ref_emb, axis=0).reshape(1, -1))

    topic_sim   = normalize(topic_sim)
    text_co_sim = normalize(text_co_sim)
    text_eu_sim = normalize(text_eu_sim)
    # --- 점수 계산 ---
    w_t, w_c, w_e = weights
    score = w_t * topic_sim + w_c * text_co_sim + w_e * text_eu_sim
    top_idx = score.argsort()[-top_k:][::-1]

    # --- 결과 구성 ---
    results = []
    for rank, i in enumerate(top_idx, 1):
        reason = generate_reason_llm(reference_docs[0], candidate_docs[i], score[i], url, headers) \
                 if use_llm else "LLM skipped"
        results.append({
            "rank": rank,
            "division": candidate_docs[i].get('division', ''),
            "title": candidate_docs[i].get('title', ''),
            "score": round(float(score[i]), 3),
            "reason": reason,
            "url": candidate_docs[i].get('url', '')
        })

    return results, summarize_topics(topic_model)
