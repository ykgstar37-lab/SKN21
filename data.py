import pandas as pd
df = pd.read_csv("C:/Users/Playdata/OneDrive - gc.gachon.ac.kr/바탕 화면/1st_data/119car.csv", encoding='cp949')


df.head()

import pandas as pd
from pathlib import Path

# 파일 경로 지정 (필요 시 경로를 수정하세요)
file_path = 'C:/Users/Playdata/OneDrive - gc.gachon.ac.kr/바탕 화면/1st_data/119car.csv'
p = Path(file_path)
print('Exists:', p.exists())

# CSV 읽기 (cp949로 읽기)
df = pd.read_csv(file_path, encoding='cp949')

# 요약 정보
print('shape:', df.shape)
display(df.head())