import os
from dotenv import load_dotenv

# .env 파일 불러오기
load_dotenv()

# ? API 키 불러오기
Data_DIR = os.getenv("DATA_DIR")
DataON_KEY = os.getenv("DATAON_KEY")
ScienceON_KEY = os.getenv("SCIENCEON_KEY")
ScienceON_clientID = os.getenv("SCIENCEON_CLIENT_ID")
ScienceON_MAC_address = os.getenv("SCIENCEON_MAC_ADDRESS")
url = os.getenv("URL")
headers = os.getenv("HEADERS")

# 검증
if not all([Data_DIR, DataON_KEY, ScienceON_KEY, ScienceON_clientID, ScienceON_MAC_address, url, headers]):
    raise ValueError("?? 환경 변수 중 누락된 항목이 있습니다. .env 파일을 확인하세요.")