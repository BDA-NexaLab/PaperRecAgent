# NexaLab 연구자 맞춤형 다국어 학술 자료 큐레이션 (추천 에이전트)
2025 데이터/AI 분석 경진대회 : 국내외 연구데이터에 대한 연관 논문·데이터 추천 에이전트 개발

## 프로젝트 개요
본 프로젝트는 KISTI 연구데이터 포털 DataON에 등록된 연구데이터를 입력으로 받아, 해당 데이터와 의미적으로 밀접한 연구논문 및 데이터셋을 자동 추천하는 AI 에이전트를 개발하는 것을 목표로 합니다.  
추천 과정은 다국어 임베딩, 토픽 기반 유사도 계산, LLM 기반 추천 사유 생성, 가중합 점수 기반 라벨링으로 구성됩니다.


## 데이터

- **수집 출처**
  - **DataON**: 국내 연구 데이터셋, REST/JSON 기반
  - **ScienceON**: 논문 정보, AES 암호화 + 토큰 + XML 처리
- **수집 필드 예시**
  - ID, 제목(title), 설명(description), 키워드(keyword), 저자(creator), 발행기관(publisher), 출판연도(year), 공개유형(public)
- **전처리**
  - HTML 태그, 특수문자 제거
  - title + description + keyword 통합
  - 한국어/영어 불용어 제거 (Okt 기반 형태소 분석 선택 적용 가능)
  - tokenization 및 normalization


## 모델

- **임베딩 모델**
  - `paraphrase-multilingual-MiniLM-L12-v2` (384차원, L2 정규화)
- **토픽 모델링**
  - `BERTopic` (nr_topics=5)
  - 임베딩 기반 군집화 (UMAP + HDBSCAN)
  - 각 토픽별 핵심 단어 및 문서-토픽 매핑 제공
- **LLM**
  - `Qwen3-14B`: 검색식 확장 및 추천 사유 생성


## 추천 점수 구성

1. **토픽 코사인 유사도**  
   - 후보 문서 토픽 벡터와 기준 문서 토픽 벡터 간 코사인 유사도
   - 후보 풀 전체에 대해 min-max 정규화(0~1)

2. **문장 임베딩 코사인 유사도**  
   - 기준 문서 임베딩 평균과 후보 문서 임베딩 간 코사인 내적

3. **임베딩 기반 유클리디안 거리 유사도**  
   - 후보 문서와 기준 문서 임베딩 간 거리 변환
   - 후보 풀 전체 min-max 정규화

- **최종 점수 계산**
```Score(c) = 0.2 * StopicNorm(c) + 0.4 * ScosNorm(c) + 0.4 * SeuclidNorm(c)```


## 추천 수준

| 수준 | 조건 |
|------|------|
| 강추 | 세 가지 신호 모두 일치, 코사인/유클리디안 ≥ 0.8 |
| 추천 | 두 가지 신호 일치, 코사인/유클리디안 ≥ 0.5 |
| 참고 | 한 가지 신호 일치, 코사인/유클리디안 ≥ 0.3 |
| 관련없음 | 코사인/유클리디안 < 0.3 |


## 학습 및 추론 수행 방법

1. **쿼리 생성**
 - 기준 문서 키워드 기반 확장형 Boolean 검색식 생성 (한글/영어)
 - 괄호 중첩, OR 연산자, 따옴표 수 제한으로 안정성 확보

2. **후보 풀 확보**
 - DataON/ScienceON API 호출, 최대 50건 수집
 - 중복 제거 및 통합

3. **임베딩 & 토픽 벡터 추출**
 - 문장 임베딩 → L2 정규화
 - BERTopic → 토픽 벡터

4. **유사도 계산**
 - 코사인 유사도, 유클리디안 기반 유사도, 토픽 코사인 유사도 계산
 - 후보 풀 전체 min-max 정규화

5. **추천 점수 통합 및 라벨링**
 - 가중합 점수 계산 후 Top-K 문서 선정
 - 추천 수준 결정 (강추/추천/참고/관련없음)

6. **추천 이유 생성**
 - Qwen3-14B LLM 활용
 - 기준 문서와 후보 문서 내용, 점수 기반으로 2~3문장 추천 사유 자동 생성

## 폴더 구조
```
NexaLab/
├── src/
│   ├── config/
│   │   └── api_keys.py
│   ├── data_collection/
│   │   ├── fetch_data.py
│   │   └── request_api.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── llm_interface/
│   │   ├── limit_boolean_query.py
│   │   ├── query_generator.py
│   │   └── reason_generator.py
│   ├── recommendation/
│   │   ├── recommender_co_eu.py
│   │   ├── recommender_cosine.py
│   │   └── recommender_euclidean.py
│   ├── utils/
│   │   ├── embedding_utils.py
│   │   ├── final_report.py
│   │   ├── similarity_utils.py
│   │   ├── text_cleaner.py
│   │   └── topic_utils.py
│   ├── weights_search.py
│   └── main.py
├── data/
│   ├── stopwords/
│   │   ├── english
│   │   └── koread
│   ├── final_test_data.csv
│   └── test_input_id.csv
├── demo/
│   └── demo영상.mp4
├── scripts/
│   └── requirements.txt
└── .env
```

## 설치 방법

1. Python 설치 (추천: Python 3.10 이상)
2. 가상환경 생성 (선택)
    ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   
3. 라이브러리 설치
    ```bash
    pip install -r requirements.txt


## 데이터
1. 수집
    ```bash
    python src/collect_data.py
2. DataOn API 호출
     ```bash
    from request_dataon import request_dataon

    query = "인공지능"
    data = request_dataon(query, row_count=10)
    print(data)
3. ScienceOn API 호출
    ```bash
    from request_scienceon import request_scienceon

    query = "자연어 처리"
    data = request_scienceon(query, row_count=10)
    print(data)


## 활용 및 기대효과

- 데이터셋 기반 연구논문/데이터 추천 자동화
- 다국어 임베딩 + 토픽 벡터 + 유사도 신호 + LLM 추천 사유 결합
- Top-K 문서 제공 및 추천 수준 라벨링
- 중저사양 환경에서도 실용적, 빠른 응답 가능
- 연구 설계, 데이터 분석 전략 수립 시 직관적 의사결정 지원