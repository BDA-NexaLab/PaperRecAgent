import numpy as np
from utils.text_cleaner import clean_text
from utils.embedding_utils import get_embeddings
from utils.topic_utils import extract_topics, summarize_topics
from utils.similarity_utils import cosine_similarity_score, combine_scores
from llm_interface.reason_generator import generate_reason_llm

def recommend_cosine_with_llm(reference_docs, candidate_docs, embed_model, okt=None, url=None, headers=None, num_topics=5, alpha=0.6, top_k=5):
    candidate_docs = [c for c in candidate_docs if c.get('title', '').strip() != reference_docs[0].get('title', '').strip()]
    if not candidate_docs:
        return [], []

    ref_texts = [clean_text(r.get('title','') + " " + r.get('description',''), okt) for r in reference_docs]
    cand_texts = [clean_text(c.get('title','') + " " + c.get('description',''), okt) for c in candidate_docs]

    all_texts = (ref_texts + cand_texts) * 10
    emb = get_embeddings(all_texts, embed_model)

    topic_model, topics, topic_vectors = extract_topics(all_texts, emb, num_topics=num_topics)
    doc_topics = topics[:len(all_texts)//10]
    unique_topics = list(set(doc_topics))

    ref_vec = np.mean([topic_vectors.get(t, np.zeros(emb.shape[1])) for t in doc_topics[:len(ref_texts)]], axis=0)
    cand_vecs = np.array([topic_vectors.get(t, np.zeros(emb.shape[1])) for t in doc_topics[len(ref_texts):]])

    ref_emb = emb[:len(ref_texts)]
    cand_emb = emb[len(ref_texts):len(ref_texts)+len(cand_texts)]

    topic_sim = cosine_similarity_score(cand_vecs, ref_vec.reshape(1, -1))
    text_sim = cosine_similarity_score(cand_emb, np.mean(ref_emb, axis=0).reshape(1, -1))
    score = combine_scores(topic_sim, text_sim, alpha)

    top_idx = score.argsort()[-top_k:][::-1]

    results = []
    for rank, i in enumerate(top_idx, 1):
        reason = generate_reason_llm(reference_docs[0], candidate_docs[i], score[i], url, headers)
        results.append({
            "rank": rank,
            "division": candidate_docs[i].get('division', ''),
            "title": candidate_docs[i].get('title', ''),
            "euclidean": round(float(score[i]), 3),
            "reason": reason
        })

    return results, summarize_topics(topic_model)
