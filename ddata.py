import pandas as pd
import sys
from pathlib import Path


def main(infile: str = 'data119.csv', outfile: str = 'data119_clean.csv') -> None:
    p = Path(infile)
    if not p.exists():
        print(f"Error: '{infile}' 파일을 찾을 수 없습니다. 현재 작업 디렉터리에서 파일 이름을 확인하세요.")
        sys.exit(1)

    # CSV 읽기: 여러 인코딩을 시도해서 읽기 실패(UnicodeDecodeError)를 방지
    def _read_csv_try_encodings(path, encodings=None):
        if encodings is None:
            encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1", "utf-16"]
        last_exc = None
        for enc in encodings:
            try:
                df_local = pd.read_csv(path, encoding=enc)
                print(f"CSV 읽기 성공 (encoding={enc})")
                return df_local, enc
            except UnicodeDecodeError as e:
                last_exc = e
                # 다음 인코딩 시도
            except Exception as e:
                # 일부 파서 에러는 인코딩 문제와 무관할 수 있으므로 기록 후 다음 시도
                last_exc = e
        # 모든 시도 실패 시 예외 발생
        raise last_exc or ValueError("파일을 읽지 못했습니다.")

    try:
        df, used_encoding = _read_csv_try_encodings(infile)
    except Exception as e:
        print("Error: CSV 파일을 읽는 중 문제가 발생했습니다:", repr(e))
        print("해결책: 1) 파일 인코딩을 확인하세요 (ex: cp949/euc-kr), 2) 필요시 'chardet' 또는 'charset-normalizer'로 인코딩을 탐지하세요.")
        sys.exit(1)

    # 간단한 요약 출력
    print("== 요약 ==")
    print("파일:", infile)
    print("행,열:", df.shape)
    print("열형식:\n", df.dtypes)
    print("\n처음 5행:\n", df.head().to_string(index=False))
    print("\n결측치 열별 합계:\n", df.isnull().sum())

    # 간단 전처리 예제:
    # 1) 중복 제거
    before = len(df)
    df = df.drop_duplicates()
    print(f"중복 제거: {before} -> {len(df)}")

    # 2) 숫자형 열의 결측치는 중앙값으로 채우기, 비숫자(객체) 열은 빈 문자열로 채우기
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna("")

    # 3) 문자열 숫자를 숫자로 변환 가능한 열은 변환 시도 (콤마 제거 포함)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # 쉼표 제거 후 숫자로 변환 시도
                cleaned = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(cleaned, errors='raise')
                print(f"열 변환: '{col}' -> numeric")
            except Exception:
                # 변환 불가 시 원래 값 유지
                pass

    # 결과 저장
    df.to_csv(outfile, index=False)
    print(f"정제된 파일을 저장했습니다: {outfile}")



if __name__ == '__main__':
    # 명령행 인자: python preprocess.py [input_csv] [output_csv]
    infile = sys.argv[1] if len(sys.argv) >= 2 else 'data119.csv'
    outfile = sys.argv[2] if len(sys.argv) >= 3 else 'data119_clean.csv'
    main(infile, outfile)
