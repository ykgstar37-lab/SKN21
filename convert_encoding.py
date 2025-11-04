"""
간단한 인코딩 변환 스크립트
사용법:
    python convert_encoding.py 입력파일.csv 출력파일.csv [입력인코딩]
예:
    python convert_encoding.py "주유소.csv" "주유소_utf8.csv" cp949

기본 동작: 입력 인코딩이 주어지면 그 인코딩으로 읽고, 없으면 여러 인코딩을 순서대로 시도합니다.
출력 인코딩은 기본으로 utf-8-sig (Excel에서 한글 깨짐 방지용 BOM 포함)를 사용합니다n"""

import argparse
from pathlib import Path
import sys

FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1", "utf-16"]


def detect_and_read(path: Path, encoding: str | None = None) -> tuple[str, str]:
    """주어진 인코딩으로 읽거나, 없으면 여러 인코딩을 시도해 읽은 텍스트와 사용된 인코딩을 반환합니다."""
    if encoding:
        try:
            text = path.read_text(encoding=encoding)
            return text, encoding
        except Exception as e:
            raise RuntimeError(f"지정한 입력 인코딩으로 파일을 읽을 수 없습니다: {encoding}: {e}")

    last_exc = None
    for enc in FALLBACK_ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
            return text, enc
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"모든 인코딩 시도 실패: {last_exc}")


def main():
    parser = argparse.ArgumentParser(description="CSV/텍스트 파일 인코딩 변환 (UTF-8로 저장)")
    parser.add_argument("infile", help="입력 파일 경로")
    parser.add_argument("outfile", help="출력 파일 경로")
    parser.add_argument("--in-enc", help="입력 파일 인코딩을 직접 지정 (예: cp949)", default=None)
    parser.add_argument("--out-enc", help="출력 인코딩 (기본: utf-8-sig)", default="utf-8-sig")
    args = parser.parse_args()

    p_in = Path(args.infile)
    p_out = Path(args.outfile)

    if not p_in.exists():
        print(f"Error: 입력 파일이 존재하지 않습니다: {p_in}")
        sys.exit(1)

    try:
        text, used = detect_and_read(p_in, encoding=args.in_enc)
    except Exception as e:
        print("파일 읽기 실패:", e)
        sys.exit(2)

    # 출력 폴더 생성
    if not p_out.parent.exists():
        p_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        p_out.write_text(text, encoding=args.out_enc)
        print(f"성공: {p_in} ({used}) -> {p_out} ({args.out_enc})")
    except Exception as e:
        print("출력 저장 실패:", e)
        sys.exit(3)


if __name__ == '__main__':
    main()
