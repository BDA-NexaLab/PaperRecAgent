import numpy as np

def get_embeddings(texts, embed_model, normalize=True):
    """문서 리스트를 임베딩"""
    return embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=normalize)
