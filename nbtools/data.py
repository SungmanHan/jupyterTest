"""실습 데이터를 **코드로** 만든다 (``data/`` 에 생성).

왜 내려받거나 커밋하지 않는가
  * 저장소에 데이터 파일을 넣지 않아도 되고, 네트워크 없이 첫 실행이 된다.
  * 원본 노트북이 참조하던 ``C:\\Users\\gridone\\Desktop\\...`` 같은 개인 PC 경로 문제가
    구조적으로 사라진다.
  * **정답을 아는 데이터**를 만들 수 있다 — 가격을 '브랜드 등급 × 연식 감가 × 주행거리'
    규칙으로 생성했으므로, 분석 결과가 그 규칙을 되찾아내는지로 스스로 채점할 수 있다.
  * 시드가 고정돼 있어 누가 언제 실행해도 같은 숫자가 나온다.

생성물
  ``data/used_cars_raw.csv``  중고차 매물 원시 데이터 (일부러 지저분하게)
  ``data/news_sample.html``   가상의 뉴스 검색 결과 페이지 (오프라인 스크래핑 실습용)
  ``data/spec_table.html``    표 하나짜리 페이지 (``pandas.read_html`` 실습용)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42

# 브랜드 → (등급 계수, 모델 목록). 계수가 가격의 '정답 규칙' 중 하나다.
BRANDS: dict[str, tuple[float, tuple[str, ...]]] = {
    "현대": (1.00, ("아반떼", "쏘나타", "그랜저", "투싼", "싼타페")),
    "기아": (0.97, ("K3", "K5", "K8", "스포티지", "쏘렌토")),
    "쉐보레": (0.85, ("스파크", "말리부", "트랙스", "이쿼녹스")),
    "르노": (0.83, ("SM6", "QM6", "XM3")),
    "제네시스": (1.85, ("G70", "G80", "GV70", "GV80")),
    "BMW": (2.10, ("320i", "520i", "X3", "X5")),
    "벤츠": (2.30, ("C200", "E250", "GLC", "GLE")),
    "테슬라": (2.05, ("모델3", "모델Y")),
}
FUELS = ("가솔린", "디젤", "LPG", "하이브리드", "전기")
FUEL_FACTOR = {"가솔린": 1.00, "디젤": 1.03, "LPG": 0.88, "하이브리드": 1.12, "전기": 1.18}

PRESSES = ("가상뉴스", "테크타임즈", "데일리리포트", "오픈데이터신문", "샘플경제")


def _used_car_rows(count: int, rng: np.random.Generator) -> list[tuple[str, str, str, str]]:
    """``(type, info, price, service)`` 튜플 목록. 원본 노트북의 컬럼 구조를 그대로 따른다."""
    rows = []
    for _ in range(count):
        brand = rng.choice(list(BRANDS))
        grade, models = BRANDS[brand]
        model = rng.choice(list(models))
        year = int(rng.integers(2015, 2025))
        month = int(rng.integers(1, 13))
        age = 2025 - year
        distance = float(np.clip(rng.normal(1.6, 0.9) * max(age, 0.5), 0.2, 22))
        # 연료는 브랜드와 독립이 아니다 — '테슬라 디젤' 같은 말이 안 되는 조합이 생기면
        # 학습자가 데이터 자체를 의심하게 된다. 지저분함은 의도한 곳(가격·공백·중복)에만 둔다.
        fuel = "전기" if brand == "테슬라" else str(rng.choice(FUELS[:4], p=[0.46, 0.27, 0.09, 0.18]))

        # 가격 = 신차가 × 연식 감가(연 12%) × 주행거리 패널티 × 연료 계수 × 노이즈
        base = 2600 * grade
        price = base * (0.88**age) * (1 - 0.018 * distance) * FUEL_FACTOR[fuel]
        price = int(max(120, price * rng.normal(1.0, 0.07)))

        info = f"{brand} {model} {year % 100:02d}/{month:02d} {distance:.1f}만km {fuel}"
        rows.append(("중고차", info, f"{price:,}만원", str(rng.choice(("무사고", "단순수리", "-")))))
    return rows


def make_used_cars(count: int = 400) -> str:
    """실제 매물 목록처럼 **지저분한** CSV 텍스트를 만든다.

    깨끗한 데이터로 연습하면 정작 실전에서 막힌다. 아래를 일부러 섞는다.
      * 가격이 숫자가 아닌 행 (``상담``, ``문의``)
      * 정보가 비어 있는 행
      * 앞뒤 공백이 붙은 값
      * 완전 중복 행
    """
    rng = np.random.default_rng(SEED)
    rows = _used_car_rows(count, rng)

    dirty: list[tuple[str, str, str, str]] = []
    for row in rows:
        kind = rng.random()
        if kind < 0.05:  # 가격 협의 매물
            dirty.append((row[0], row[1], str(rng.choice(("상담", "문의"))), row[3]))
        elif kind < 0.08:  # 정보 누락
            dirty.append((row[0], "", row[2], row[3]))
        elif kind < 0.13:  # 공백 오염
            dirty.append((row[0], f"  {row[1]} ", f" {row[2]}", row[3]))
        else:
            dirty.append(row)
        if rng.random() < 0.02:  # 중복 게시
            dirty.append(dirty[-1])

    header = "type,info,price,service"
    lines = [header] + [f'{t},"{i}","{p}",{s}' for t, i, p, s in dirty]
    return "\n".join(lines) + "\n"


def make_news_html() -> str:
    """가상의 뉴스 검색 결과 페이지.

    실제 포털 HTML 을 복사하지 않는다 — 구조만 비슷하게 만든 **가짜 페이지**다.
    실습이 특정 사이트의 DOM 변경이나 이용약관에 묶이지 않게 하기 위함이다.
    (원본 노트북이 쓰던 ``div.thumb`` 선택자는 지금 네이버에 존재하지 않는다.)
    """
    rng = np.random.default_rng(SEED + 1)
    topics = [
        ("RPA 도입 기업 늘어", "제조업 중심으로 단순 반복 업무 자동화가 확산되고 있다."),
        ("사무 자동화 시장 성장", "국내 사무 자동화 시장이 전년 대비 성장했다는 조사 결과가 나왔다."),
        ("업무 자동화와 일자리", "자동화가 일자리에 미치는 영향을 두고 논의가 이어지고 있다."),
        ("오픈소스 자동화 도구 비교", "무료로 쓸 수 있는 자동화 도구들의 특징을 정리했다."),
        ("자동화 도입 실패 사례", "목표 없이 도입한 자동화는 오히려 비용을 늘린다는 지적이다."),
        ("문서 처리 자동화", "OCR 과 규칙 기반 처리를 결합한 사례가 소개됐다."),
    ]
    items = []
    for i in range(18):
        title, summary = topics[i % len(topics)]
        press = PRESSES[i % len(PRESSES)]
        # 같은 기사가 두 번 실린 경우 — 중복 제거 실습용
        article_id = 1000 + (i if i not in (5, 11) else 3)
        items.append(f"""
    <li class="news-item">
      <a class="news-title" href="/article/{article_id}">{title} ({i + 1}회차)</a>
      <div class="news-meta">
        <span class="press">{press}</span>
        <span class="date">2025-0{i % 9 + 1}-{i % 27 + 1:02d}</span>
        <span class="views">{int(rng.integers(100, 9000))}</span>
      </div>
      <p class="news-summary">{summary}</p>
    </li>""")

    return f"""<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"><title>검색 결과 - 업무 자동화</title></head>
<body>
  <header><h1>가상 뉴스 검색</h1></header>
  <main>
    <p class="result-count">총 <strong>18</strong>건</p>
    <ul class="news-list">{"".join(items)}
    </ul>
    <nav class="paging">
      <a href="?page=1" class="current">1</a>
      <a href="?page=2">2</a>
      <a href="?page=3">3</a>
    </nav>
  </main>
</body>
</html>
"""


def make_spec_table_html() -> str:
    """``pandas.read_html`` 실습용 표 페이지 (숫자에 쉼표·단위가 섞여 있다)."""
    rows = "".join(
        f"<tr><td>{name}</td><td>{seats}</td><td>{fuel}</td><td>{price}</td></tr>"
        for name, seats, fuel, price in [
            ("아반떼", 5, "가솔린", "1,980만원"),
            ("쏘렌토", 7, "디젤", "3,450만원"),
            ("모델Y", 5, "전기", "5,690만원"),
            ("GV80", 5, "가솔린", "7,120만원"),
            ("스파크", 4, "가솔린", "1,180만원"),
        ]
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>차량 제원</title></head>
<body>
  <h1>차량 제원 표</h1>
  <table id="spec">
    <thead><tr><th>모델</th><th>승차정원</th><th>연료</th><th>신차가격</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body></html>
"""


_MAKERS = {
    "used_cars_raw.csv": make_used_cars,
    "news_sample.html": make_news_html,
    "spec_table.html": make_spec_table_html,
}


def ensure_data(force: bool = False, quiet: bool = False) -> Path:
    """없는 실습 데이터만 생성하고 ``data/`` 경로를 돌려준다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, maker in _MAKERS.items():
        path = DATA_DIR / name
        if force or not path.exists():
            path.write_text(maker(), encoding="utf-8")
            if not quiet:
                print(f"생성: data/{name}")
    return DATA_DIR


def load_used_cars(**kwargs):
    """원시 CSV 를 그대로 읽어 온다 (정제는 노트북에서 직접 한다)."""
    import pandas as pd

    ensure_data(quiet=True)
    return pd.read_csv(DATA_DIR / "used_cars_raw.csv", **kwargs)


if __name__ == "__main__":
    print(f"데이터 준비 완료 → {ensure_data()}")
