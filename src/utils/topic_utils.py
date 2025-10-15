from bertopic import BERTopic
import numpy as np
from utils.text_cleaner import korean_stopwords, all_english_stopwords

# 토픽 추출
def extract_topics(texts, embeddings, num_topics=5):
    topic_model = BERTopic(nr_topics=num_topics, calculate_probabilities=False)
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

    topic_vectors = {}
    for t in set(topics):
        if t == -1:
            continue
        idx = [i for i, top in enumerate(topics) if top == t]
        topic_vectors[t] = np.mean(embeddings[idx], axis=0)
    return topic_model, topics, topic_vectors

# 토픽 요약
def summarize_topics(topic_model):
    topic_info = topic_model.get_topics()
    summaries = []
    for topic_num, words in topic_info.items():
        if topic_num == -1:
            continue
        filtered_words = [w for w, _ in words if w.lower() not in all_english_stopwords and w not in korean_stopwords]
        if filtered_words:
            summaries.append(f"Topic {topic_num}: {', '.join(filtered_words[:5])}")
    return summaries