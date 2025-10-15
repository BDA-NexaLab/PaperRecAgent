import requests

def generate_query(input_data, url,headers):
  # 요청 데이터
  data = {
      "model": "qwen3-14b",
      "messages": [
          {"role": "system","content":f'''
당신은 학술 정보 검색 전문가입니다. 사용자가 제공한 검색 키워드를 기반으로 국내외 학술 DB에서 최대한 많은 관련 자료를 찾을 수 있는 **확장형 Boolean 검색식**을 만들어야 합니다.

조건:
1. 핵심 개념과 동의어를 파악하고, {input_data[0]['keyword']}를 반영합니다.
2. 너무 구체적이거나 제외 조건(-)은 가능한 한 제거하여 자료 누락을 방지합니다.
3. 국내용 검색식(Korean)과 국제용 검색식(English)을 각각 만듭니다.
4. Boolean 연산자 사용법:
   - () : 높은 우선 순위
   - 공백 : AND
   - | : OR
   - "" : 정확히 일치하는 문구 검색

출력 형식(항상 2개):
Korean : [검색식]
English : [검색식]

추가 지침:
- AND, OR 문자를 사용하지 말고 검색 연산자 공백과 |을 사용하세요.
- 동의어나 유사 표현을 | 로 묶습니다.
- 핵심 개념을 중심으로 공백으로 연결합니다.
- 불필요한 제외 조건은 제거하고, 너무 좁은 의미의 검색식은 확장합니다.
- 자료가 없더라도 반드시 2개의 출력값을 제공합니다.
- 연도를 나타내는 숫자는 제외합니다.
- 불용어와 불필요한 단어는 제외하고 키워드 적극적으로 반영하여 생성합니다.
- ()는 최대 3개만, |는 최대 9개만, ""는 최대 2개만 사용하세요.

예시:
['(multi type model | multi-type model) (birth death | birth-death) (bayesian inference | bayesian phylogenetics)']
'''
          },
          {"role": "user", "content": f"""
                                          Title: {input_data[0]['title']}
                                          Topic: {input_data[0]['mnsb_pc']}
                                          Description: {input_data[0]['description']}
                                          Keywords: {input_data[0]['keyword']}
                                          """
          }

      ],
      "max_tokens": 1000,  # 충분히 길게 설정
      "temperature": 0.7
  }

  # API 호출
  response = requests.post(url, headers=headers, json=data)

  # 결과 출력
  if response.status_code == 200:
      result = response.json()
      assistant_text = result['choices'][0]['message']['content']

      # 문자열에서 각 검색식 추출
      lines = assistant_text.strip().split("\n")
      korean_queries = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Korean")]
      english_queries = [line.split(":", 1)[1].strip() for line in lines if line.startswith("English")]

      print(f"한글 검색식: {korean_queries}")
      print(f"영어 검색식: {english_queries}")

      return korean_queries, english_queries

  else:
      print("Error:", response.status_code, response.text) # 24초
