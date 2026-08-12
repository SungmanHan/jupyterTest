"""``python -m nbtools`` — 실습 데이터를 생성한다.

    python -m nbtools            # 없는 것만 생성
    python -m nbtools --force    # 전부 다시 생성
"""

import sys

from nbtools.data import ensure_data

path = ensure_data(force="--force" in sys.argv, quiet=False)
print(f"데이터 준비 완료 → {path}")
