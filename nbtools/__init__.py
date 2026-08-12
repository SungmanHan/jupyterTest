"""노트북 공용 유틸 — 실습 데이터 생성과 표시 헬퍼.

노트북마다 반복되는 준비 작업(데이터 만들기, 한글 폰트 설정, OpenCV 이미지 인라인 표시)만
모아 둔 얇은 패키지다. 학습 대상인 pandas·matplotlib·BeautifulSoup 호출 자체는
노트북 안에서 직접 한다.
"""

from nbtools.data import DATA_DIR, ROOT, ensure_data, load_used_cars
from nbtools.display import imshow, use_korean_font

__all__ = ["DATA_DIR", "ROOT", "ensure_data", "imshow", "load_used_cars", "use_korean_font"]
