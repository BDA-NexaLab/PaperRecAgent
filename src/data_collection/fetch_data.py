from data_collection.request_api import request_dataon, request_scienceon
import re

def collect_data(query, category_api, row_count=50, return_type="str"):
    import re

    # 데이터 가져오기
    if category_api == "DataOn":
        data = request_dataon(query, row_count=row_count)
        desired_keys = ["svc_id", "title", "mnsb_pc", "description", "keyword", "url"]
        division = "dataset"
    elif category_api == "ScienceOn":
        data = request_scienceon(query, row_count=row_count)
        desired_keys = ["title", "description", "keyword", "url"]
        division = "paper"
    else:
        print(f"Unknown category_api: {category_api}")
        return []

    if not data:
        print(f"'{query}' 검색 결과가 없습니다.")
        return []

    processed_docs = []

    for doc in data:
        field_texts = []  # 문자열 합치기용
        processed_doc = {}
        for k in desired_keys:
            text = doc.get(k, "") or ""

            # 리스트이면 쉼표로 합치기
            if isinstance(text, list):
                text = ", ".join(filter(None, text))

            # HTML 태그 제거
            text = re.sub(r"<[^>]*>", " ", text)

            # \r, \n 제거
            text = text.replace("\\r", " ").replace("\\n", " ").replace("\r", " ").replace("\n", " ")

            # 특수문자 제거 (문자+숫자+공백+쉼표만 남김)
            text = re.sub(r"[^가-힣a-zA-Z0-9\s,]", " ", text)

            # 연속 공백 제거
            text = re.sub(r"\s+", " ", text).strip()

            processed_doc[k] = text
            field_texts.append(text)

             # division 추가
            processed_doc["division"] = division

        # 최종 반환 형태 결정
        if return_type == "str":
            combined_text = ", ".join(f"{k}: {text}" for k, text in zip(desired_keys, field_texts))
            processed_docs.append(combined_text)
        else:
            processed_docs.append(processed_doc)

    return processed_docs
