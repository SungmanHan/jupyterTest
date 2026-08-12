# jupyterTest — 주피터 노트북으로 배우는 데이터 작업

> Jupyter · NumPy · pandas · matplotlib · 웹 스크래핑 · 노트북에서의 OpenCV.
> **순서대로 따라가는 노트북 8권**. 데이터는 코드로 생성되므로 네트워크 없이 바로 실행된다.

2019년의 실험 노트북 16개(`Untitled.ipynb` ~ `Untitled6.ipynb` 포함)와 스크립트 4개에서 출발했다.
원본은 전부 [`legacy/`](legacy/) 에 보존했고 — **대부분 지금 환경에서는 실행되지 않는다** —
무엇이 왜 바뀌었는지는 [`legacy/README.md`](legacy/README.md) 에 정리했다.

---

## 빠른 시작

```bash
git clone https://github.com/SungmanHan/jupyterTest.git
cd jupyterTest
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m nbtools    # 실습 데이터 생성 (노트북 안에서도 자동 생성된다)
jupyter lab          # notebooks/ 를 열고 01부터
```

VS Code·PyCharm 에서 열어도 된다. 그 경우 `jupyterlab` 대신 `ipykernel` 만 있으면 된다.

---

## 학습 순서

| # | 노트북 | 답할 수 있게 되는 질문 |
|:--:|---|---|
| 01 | [Jupyter 기본기](notebooks/01_jupyter_basics.ipynb) | 왜 같은 코드가 다른 결과를 내는가? `pip install` 했는데 왜 `ModuleNotFoundError` 인가? |
| 02 | [NumPy 핵심](notebooks/02_numpy_essentials.ipynb) | 브로드캐스팅은 어떤 규칙인가? 루프 대신 벡터화하면 몇 배 빨라지는가? |
| 03 | [pandas 핵심](notebooks/03_pandas_essentials.ipynb) | `.loc` 과 `.iloc` 은 왜 결과 개수가 다른가? `append` 는 어디로 갔는가? |
| 04 | [데이터 정제 실전](notebooks/04_data_cleaning.ipynb) | `"1,850만원"` 이 섞인 컬럼을 어떻게 숫자로 만드는가? 정제가 맞았는지 어떻게 확인하는가? |
| 05 | [시각화](notebooks/05_visualization.ipynb) | 한글이 왜 □ 로 나오는가? 막대그래프에 원자료를 그대로 넣으면 무슨 일이 생기는가? |
| 06 | [웹 스크래핑 기초](notebooks/06_web_scraping_basics.ipynb) | HTML 에서 표를 뽑아 CSV 로. robots.txt 는 어떻게 확인하는가? |
| 07 | [동적 페이지와 Selenium](notebooks/07_dynamic_pages_selenium.ipynb) | 브라우저 자동화가 정말 필요한가? 2019년 Selenium 코드는 왜 안 도는가? |
| 08 | [주피터에서 OpenCV](notebooks/08_opencv_in_jupyter.ipynb) | `cv2.imshow` 는 왜 커널을 멈추는가? |

04·05·06 은 원본 노트북이 하려던 작업(중고차 데이터 분석, 뉴스 수집)을 **같은 목표, 다른 방법**으로 다시 쓴 것이다.

---

## 저장소 구조

```
jupyterTest/
├── notebooks/     학습 노트북 8권 — 번호 순서대로
├── nbtools/       공용 유틸 (실습 데이터 생성, 한글 폰트, OpenCV 인라인 표시)
├── legacy/        2019년 원본 + 무엇이 왜 바뀌었는지
├── docs/          설치 문제 해결, 노트북 위생 가이드
├── data/          생성되는 실습 데이터 (git 제외)
└── outputs/       노트북 실행 산출물 (git 제외)
```

### 실습 데이터를 코드로 만드는 이유

`data/` 의 CSV·HTML 은 [`nbtools/data.py`](nbtools/data.py) 가 생성한다.

* 원본이 참조하던 `C:\Users\gridone\Desktop\...` 같은 **개인 PC 경로 문제가 구조적으로 사라진다**
* 저장소에 데이터 파일을 커밋하지 않아도 되고, 네트워크 없이 첫 실행이 된다
* **정답 규칙을 아는 데이터**다 — 중고차 가격을 `브랜드 등급 × 연식 감가 × 주행거리` 로 생성했기 때문에,
  04 노트북은 분석 결과가 그 규칙을 되찾는지로 **자기 채점**한다 (상관계수 출력)
* 시드가 고정돼 있어 누가 언제 실행해도 같은 숫자가 나온다

### 노트북 출력을 커밋하는 이유

일반적으로는 `.ipynb` 출력을 커밋하지 않는 것이 정석이다(diff 폭발). 이 저장소는 **학습 자료**라
GitHub 에서 바로 결과를 볼 수 있는 편이 낫다고 판단해 출력을 포함한다. 대신:

* 모든 노트북은 **Restart & Run All** 로 위에서부터 한 번에 실행된 상태다 (셀 번호 1,2,3… 순서)
* 난수 시드를 고정해 실행할 때마다 같은 결과가 나온다

출력 없이 관리하고 싶다면 `pip install nbstripout && nbstripout --install` 한 번이면 된다.
(01 노트북 §5에서 다룬다)

---

## 2019 → 지금, 무엇이 달라졌나

원본 노트북이 지금 실행되지 않는 주된 이유들이다. 각각은 해당 노트북에서 실제 코드로 확인한다.

| 변경 | 영향 | 다루는 곳 |
|---|---|---|
| `DataFrame.append` **삭제** (pandas 2.0) | 원본의 행 누적 루프가 `AttributeError` | 03 §5, 04 |
| ragged `np.array()` **예외** (NumPy 1.24) | `np.array(contours)` 가 `ValueError` | 02 §8, 08 §3 |
| Selenium 4 — 드라이버 경로 인자·`find_element_by_*` **삭제** | `webdriver.Chrome('C:/App/...')` 가 실패 | 07 §2 |
| `cv2.CascadeClassifier` 제거 (OpenCV 5) | 옛 얼굴 검출 예제 실행 불가 | 08, opencvStudy |
| 포털 DOM 변경 | `div.thumb` 선택자가 0건 반환 | 06 |

---

## 더 보기

* [docs/setup.md](docs/setup.md) — 커널·가상환경 불일치, 한글 폰트, 설치 문제
* [docs/notebook-hygiene.md](docs/notebook-hygiene.md) — 노트북 버전관리·리뷰·재현성
* [legacy/README.md](legacy/README.md) — 원본 20개가 지금 왜 안 도는가 (실측 오류 메시지 포함)
* [opencvStudy](https://github.com/SungmanHan/opencvStudy) — 이미지 처리 자체는 이쪽에 레슨 16개로 정리
