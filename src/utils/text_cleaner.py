import re
from konlpy.tag import Okt

okt = Okt()

# === 영어 불용어 ===
english_stopwords_path = "data/stopwords/english"

with open(english_stopwords_path, 'r', encoding='utf-8') as f:
    english_stopwords = set([line.strip().lower() for line in f if line.strip()])

paper_stopwords_en = {
    'study', 'research', 'result', 'results', 'method', 'methods',
    'data', 'analysis', 'based', 'using', 'approach', 'show', 'shown',
    'paper', 'model', 'models', 'effect', 'effects', 'provide', 'at'
}

all_english_stopwords = english_stopwords.union(paper_stopwords_en)

# === 한국어 불용어 ===
korean_stopwords_path = "data/stopwords/koread"

with open(korean_stopwords_path, 'r', encoding='utf-8') as f:
    korean_file_stopwords = set([line.strip() for line in f if line.strip()])

# 논문 특화 한국어 불용어 하드코딩
paper_stopwords_ko = {
    '의', '가', '이', '은', '는', '을', '를', '에', '으로', '와', '과',
    '도', '에서', '하다', '이다', '있다', '되다', '및', '하지만', '또한',
    '그러나', '때문에', '그', '저', '등', '연구', '데이터', '방법',
    
    '실험', '결과', '분석', '연구자', '방법론', '제안', '논문',
    '사용', '데이터셋', '관찰', '평가', '측정', '제시', '모형', '기반'
}
korean_stopwords = korean_file_stopwords
all_korean_stopwords = korean_file_stopwords.union(paper_stopwords_ko)

# === 텍스트 전처리 함수 ===
# utils/text_cleaner.py
import re

def clean_text(text: str, okt=None) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^가-힣a-z\s]", " ", t)

    tokens = None
    if okt is not None:
        try:
            tokens = okt.morphs(t)
        except Exception:
            tokens = None

    if tokens is None:
        try:
            from konlpy.tag import Okt
            _okt = Okt()
            tokens = _okt.morphs(t)
        except Exception:
            tokens = t.split()

    tokens = [w for w in tokens if w not in all_korean_stopwords and w not in all_english_stopwords]
    return " ".join(tokens)


def prepare_query(q: str, max_terms: int = 8) -> str:
    """
    LLM이 만든 'A | B | C' 형식의 검색식을 보존하되, 과도한 토큰은 줄여준다.
    - |, (), 따옴표는 유지
    - 중복/공백 정리
    - 너무 긴 경우 상위 N개로 자르기
    """
    if not q:
        return ""

    # 파이프가 있다면 파이프 기준으로 분해해서 top-N만
    parts = [p.strip() for p in q.split("|") if p.strip()]
    if parts:
        parts = parts[:max_terms]
        # 공백이 포함된 구는 따옴표로 감싸주면 매칭률↑
        parts = [p if p.startswith('"') or p.endswith('"') or " " not in p else f'"{p}"' for p in parts]
        return " OR ".join(parts)

    # 파이프가 없다면: 괄호/따옴표 유지, 나머지만 가볍게 정리
    keep = re.sub(r"[^\w\s\-\(\)\"']", " ", q)  # (), ", ' 는 유지
    tokens = [t for t in keep.split() if t]
    tokens = tokens[:max_terms]
    return " ".join(tokens)
