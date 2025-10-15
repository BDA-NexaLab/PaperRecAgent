import requests
import re

def generate_reason_llm(reference_text, candidate_text, similarity_score, url, headers):
      data = {
        "model": "qwen3-14b",
        "messages": [
            {"role": "system", "content": f"""
당신은 학술 데이터 및 논문 추천 시스템의 설명 생성자입니다.
오직 최종 추천 이유 문장만 출력하십시오.

두 문서의 의미적/주제적 유사도 점수는 {similarity_score:.3f}입니다.

다음 기준으로 추천 이유를 생성하세요:
1. 주제 유사성: 두 논문의 연구 주제, 연구 대상, 주요 개념을 비교하여 설명
2. 연구 방법 유사성: 사용된 연구 방법론, 분석 기법, 데이터 처리 방식을 비교
3. 적용 분야 유사성: 연구가 적용되는 분야, 활용 가능성, 문제 해결 영역 등을 명확히 서술

출력 지침:
- 문체: 공식적이고 자연스러운 한국어
- 문장 수: 정확히 2~3문장
- 표현: '~등'이나 모호한 단어 사용 금지
- 형식: 반드시 아래 형태로 출력
"추천 이유: [주제 유사성]. [연구 방법 유사성]. [적용 분야 유사성]."
            """},
            {"role": "user", "content": f"""
                                            기준 문서: {reference_text}
                                            후보 문서: {candidate_text}
                                        """
            }
        ],
        "max_tokens": 800,
        "temperature": 0.7
      }

      try:
          response = requests.post(url, headers=headers, json=data)
          resp_json = response.json()

          # 디버깅 출력
          if 'error' in resp_json:
              print(f"LLM 에러: {resp_json['error']}")
              return "추천 이유 생성 실패"

          text = resp_json['choices'][0]['message']['content'].strip()
          text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

          if not text:
              print("빈 응답 받음")
              return "추천 이유 생성 실패"

          return text
      except requests.exceptions.Timeout:
        print("LLM 호출 타임아웃")
        return "타임아웃 오류"
      except requests.exceptions.HTTPError as e:
          print(f"HTTP 에러: {e.response.status_code}")
          return "HTTP 오류"
      except Exception as e:
          print(f"LLM 호출 오류: {type(e).__name__}: {e}")
          return "추천 이유 생성 실패"
