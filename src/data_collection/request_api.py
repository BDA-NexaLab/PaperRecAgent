import requests, xml.etree.ElementTree as ET, json
from urllib import parse
import re, datetime, traceback
from Cryptodome.Cipher import AES # type: ignore
import base64
from config.api_keys import (
    DataON_KEY,
    ScienceON_KEY,
    ScienceON_clientID,
    ScienceON_MAC_address
)

def request_dataon(search_query, row_count=1):
    BASE_SEARCH_URL = "https://dataon.kisti.re.kr/rest/api/search/dataset"

    all_results = []
    params = {
        "key": DataON_KEY,
        "query": search_query,
        "from": 0,
        "size": row_count
    }
    response = requests.get(BASE_SEARCH_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        if not records:
            print(f"No records found for '{search_query}'")

        for rec in records:
            all_results.append({
                "query": search_query,
                "svc_id": rec.get("svc_id", ""),
                "mnsb_pc" : rec.get("dataset_mnsb_pc", ""),
                "title" : rec.get("dataset_title_etc_main"),
                "description": rec.get("dataset_expl_etc_main"),
                "keyword": rec.get("dataset_kywd_etc_main"),
                "creator": rec.get("dataset_creator_etc_sub", ""),
                "publisher": rec.get("cltfm_etc", ""),
                "year": rec.get("dataset_pub_dt_pc", ""),
                "public": rec.get("dataset_access_type_pc", ""),
                "url": rec.get("dataset_lndgpg", "")  # 데이터 제공처 링크
            })
    else:
        print(f"Error searching '{search_query}': {response.status_code}, {response.text}")

    return all_results

class AESTestClass:
    def __init__(self, plain_txt, key):
        # iv, block_size 값은 고정입니다.
        self.iv = 'jvHJ1EFA0IXBrxxz'
        self.block_size = 16
        self.plain_txt = plain_txt
        self.key = key

    def pad(self):
        number_of_bytes_to_pad = self.block_size - len(self.plain_txt) % self.block_size
        ascii_str = chr(number_of_bytes_to_pad)
        padding_str = number_of_bytes_to_pad * ascii_str
        print(padding_str.encode('utf-8'))
        padded_plain_text = self.plain_txt + padding_str
        return padded_plain_text

    def encrypt(self):
        cipher = AES.new(self.key.encode('utf-8'), AES.MODE_CBC, self.iv.encode('utf-8'))
        padded_txt = AESTestClass.pad(self)
        encrypted_bytes = cipher.encrypt(padded_txt.encode('utf-8'))
        encrypted_str = base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")
        return encrypted_str

refreshToken = None
accessToken = None

def createToken():
    global refreshToken, accessToken
    try:
        time = ''.join(re.findall(r"\d", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        plain_txt = json.dumps({"datetime": time, "mac_address": ScienceON_MAC_address}).replace(" ", "")
        encryption = AESTestClass(plain_txt, ScienceON_KEY)
        encrypted_txt = encryption.encrypt()

        target_URL = f"https://apigateway.kisti.re.kr/tokenrequest.do?client_id={ScienceON_clientID}&accounts={encrypted_txt}"
        response = requests.get(target_URL)
        response.raise_for_status()

        json_object = response.json()
        refreshToken = json_object.get('refresh_token')
        accessToken = json_object.get('access_token')
        print('새 Refresh Token과 Access Token 발급 완료')
    except Exception:
        traceback.print_exc()

def getAccessToken():
    global accessToken
    try:
        target_URL = f"https://apigateway.kisti.re.kr/tokenrequest.do?refreshToken={refreshToken}&client_id={ScienceON_clientID}"
        response = requests.get(target_URL)
        response.raise_for_status()
        if 'errorCode' in response.text:
            return None
        json_object = response.json()
        accessToken = json_object.get('access_token')
        print('Access Token 재발급 완료')
        return accessToken
    except Exception:
        traceback.print_exc()
        return None

def request_scienceon(search_query, row_count=3, retry=0):
    global accessToken
    MAX_RETRY = 2

    if not accessToken:
        createToken()

    query = {"BI": search_query}
    search_query_encoded = parse.quote(json.dumps(query, ensure_ascii=False))
    target_URL = (
        f"https://apigateway.kisti.re.kr/openapicall.do?"
        f"client_id={ScienceON_clientID}&token={accessToken}&version=1.0&action=search&target=ARTI"
        f"&searchQuery={search_query_encoded}&curPage=1&rowCount={row_count}"
    )

    try:
        response = requests.get(target_URL)
        response.raise_for_status()
    except Exception as e:
        print("HTTP 요청 실패:", e)
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print("XML 파싱 실패:", response.text)
        return None

    status_elem = root.find('resultSummary/statusCode')
    if status_elem is None:
        print("응답에 statusCode 없음:", response.text)
        return None

    statusCode = status_elem.text
    if int(statusCode) != 200:
        errorCode_elem = root.find('errorDetail/errorCode')
        errorMessage_elem = root.find('errorDetail/errorMessage')
        errorCode = errorCode_elem.text if errorCode_elem is not None else "Unknown"
        errorMessage = errorMessage_elem.text if errorMessage_elem is not None else "Unknown"

        if errorCode == 'E4103' and retry < MAX_RETRY:  # 토큰 만료
            print("AccessToken 만료, 재발급 시도...")
            if not getAccessToken():
                createToken()
            return request_scienceon(search_query, row_count, retry + 1)
        else:
            print(f"알 수 없는 오류 발생: {errorMessage}")
            return None

    # XML → 리스트 변환
    records = []
    for record in root.findall(".//record"):
        title_elem = record.find(".//item[@metaCode='Title']")
        description_elem = record.find(".//item[@metaCode='Abstract']")
        keyword_elem = record.find(".//item[@metaCode='Keyword']")
        url_elem = record.find(".//item[@metaCode='FulltextURL']")

        records.append({
            "title": title_elem.text if title_elem is not None else "",
            "description": description_elem.text if description_elem is not None else "",
            "keyword": keyword_elem.text if keyword_elem is not None else "",
            "url": url_elem.text if url_elem is not None else ""

        })

    return records
