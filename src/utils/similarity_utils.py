import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def euclidean_similarity(vec_a, vec_b):
    return 1 / (1 + euclidean_distances(vec_a, vec_b).flatten())

def cosine_similarity_score(vec_a, vec_b):
    return cosine_similarity(vec_a, vec_b).flatten()

def combine_scores(topic_sim, text_sim, alpha=0.6):
    return alpha * topic_sim + (1 - alpha) * text_sim
