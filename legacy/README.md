# legacy — 2019년 원본 노트북 (보존용)

이 저장소의 출발점이 된 파일들이다. **삭제하지 않는 이유**는 "무엇이 왜 바뀌었는지" 비교할 수
있어야 학습 자료로서 의미가 있기 때문이다. 실행은 권장하지 않는다 — 대부분 지금 환경에서 멈춘다.

## 원본 → 지금

| 원본 | 내용 | 대체 |
|---|---|---|
| `Untitled.ipynb` | 중고차 CSV 파싱 → 차트 → PDF 리포트 | [04 데이터 정제](../notebooks/04_data_cleaning.ipynb), [05 시각화](../notebooks/05_visualization.ipynb) |
| `News screpion.ipynb` | Selenium 으로 포털 뉴스 링크 수집 → CSV | [06 스크래핑 기초](../notebooks/06_web_scraping_basics.ipynb), [07 Selenium](../notebooks/07_dynamic_pages_selenium.ipynb) |
| `cv2 test.ipynb`, `cv2 test 1.ipynb`, `cv2 tes 2.ipynb` | 컨투어로 이미지 자동 크롭 | [08 주피터에서 OpenCV](../notebooks/08_opencv_in_jupyter.ipynb) + [opencvStudy](https://github.com/SungmanHan/opencvStudy) |
| `Untitled3~6.ipynb`, `근사화.ipynb`, `영역 크기.ipynb` | 컨투어·approxPolyDP 반복 실험 (내용이 서로 거의 같음) | 위와 동일 |
| `image_marsking.ipynb`, `ImageMasking.py` | 사각형으로 영역 가리기 | opencvStudy `projects/redact.py` |
| `이미지 회전.ipynb`, `image 끝 기준으로 자르기.ipynb` | 회전·크롭 | opencvStudy 레슨 04 |
| `Untitled1.ipynb` | 파일명 문자열 조작 실험 | opencvStudy `projects/auto_crop.py` |
| `Untitled2.ipynb` | 노트북 안에서 `argparse` (노트북에서는 동작하지 않는다) | — |
| `20191119/` | 크롤링 결과 CSV (본문 없이 URL 만) | 06 노트북이 `outputs/scrap/` 에 생성 |

---

## 1. `Untitled.ipynb` — 중고차 분석

### 실측 실패 (pandas 3.0.5)

```python
new_data = new_data.append(pd.Series(dataSplit(temp), index=new_data.columns), ignore_index=True)
# AttributeError: 'DataFrame' object has no attribute 'append'
```

`DataFrame.append` 는 pandas 1.4 에서 deprecate, **2.0 에서 삭제**됐다.
(확인: `hasattr(pd.DataFrame, "append")` → `False`)

### 그 밖의 문제

| # | 원본 | 문제 | 지금 |
|---|---|---|---|
| 1 | `pd.read_csv("C:\\Users\\gridone\\Desktop\\차량 정보\\...")` | 개인 PC 경로 — 아무도 재현할 수 없다 | `nbtools/data.py` 가 생성 |
| 2 | 60줄짜리 `dataSplit()` 문자열 인덱스 파싱 | 읽기·수정 불가 | 정규식 명명 그룹 6줄 |
| 3 | `vData.find("19/")` | **2019년식만** 인식. 2020년 매물은 조용히 누락 | `(?P<year2>\d{2})/(?P<month>\d{2})` |
| 4 | 브랜드를 `if vData.find("쉐보레") > -1 or ...` 로 나열 | 목록에 없는 브랜드는 빈 값 | 패턴이 자동 추출 |
| 5 | 루프 안 `append` | O(n²) + 2.0에서 삭제 | 벡터화된 `str.extract` |
| 6 | `plt.title('vBrand and vPrice')` | 한글 폰트 미설정이라 변수명을 그대로 노출 | `use_korean_font()` |
| 7 | `plt.bar(work_data['vBrand'], work_data['vPrice'])` | 집계 없이 400행을 막대로 → 겹쳐 그려짐 | `groupby().mean()` 후 정렬 |
| 8 | 검증 없음 | 정제가 맞았는지 알 수 없다 | `assert` 불변조건 + 생성 규칙과 상관 대조 |

---

## 2. `News screpion.ipynb` — 뉴스 수집

### 지금 실행되지 않는 이유

```python
driver = webdriver.Chrome('C:/App/chromedriver_win32/chromedriver')
```

Selenium 4 에서 드라이버 경로를 위치 인자로 넘기는 형태는 사라졌다(`executable_path` 는 4.10 에서 삭제,
첫 위치 인자는 이제 `options`). 지금은 **경로 지정 자체가 불필요**하다 — Selenium Manager 가
드라이버를 자동으로 맞춰 준다.

### 숨은 버그 세 가지 (당시에도 버그였다)

**① `'__file__'` 이 문자열이다**

```python
base_dir = os.path.dirname(os.path.realpath('__file__')).replace("jupyter_workspace","scrap")
```

노트북에는 `__file__` 변수가 없어 따옴표로 감싼 것인데, 결과는 이렇게 된다.

```
realpath('__file__')  →  /현재폴더/__file__
dirname(...)          →  /현재폴더          ← 그냥 현재 폴더
.replace(...)         →  /현재폴더          ← 폴더명에 jupyter_workspace 가 없으면 무효
```

즉 `scrap` 폴더 분리는 **폴더 이름이 우연히 맞을 때만** 동작했다. 지금은 `pathlib` 로 명시한다.

**② 함수가 인자 대신 전역 변수를 읽는다**

```python
def FileCheck(file, arr):
    if os.path.isfile(file) == False: ...
    else:
        data = pd.read_csv(filePaht)   # ← 인자 file 이 아니라 전역 filePaht (오타 변수)
```

호출 순서가 바뀌면 엉뚱한 파일을 읽는다. 노트북에서 전역 변수를 함수 안에서 참조할 때 생기는 전형적인 사고다.

**③ 루프마다 `data.loc[len(data)] = temp`**

행을 하나씩 붙일 때마다 전체를 다시 만든다(O(n²)). 리스트에 모아 마지막에 `pd.DataFrame(rows)` 한 번이 정답이다.

### 구조적 문제

| 문제 | 지금 |
|---|---|
| 특정 포털의 2019년 DOM(`div.thumb`)에 의존 | 선택자를 인자로 분리 + 로컬 샘플 HTML 로 실습 |
| 정적 페이지인데 Selenium 사용 | requests + BeautifulSoup 로 충분한지 먼저 판별 |
| `driver.quit()` 없음 | `contextmanager` 로 보장 |
| 대기 로직 없음 | `WebDriverWait` 조건 대기 |
| 타임아웃·재시도 없음 | 백오프 재시도 |
| robots.txt·요청 간격·약관 고려 없음 | 06 노트북 §8 체크리스트 |
| 검색어에 회사·경쟁사명이 하드코딩 | 인자로 |

> 선택자가 맞지 않아도 `find_all` 은 **예외 없이 빈 리스트**를 준다. 그래서 이런 코드는
> "에러 없이 0건" 으로 조용히 실패한다 — 스크래퍼에 **수집 건수 검증**이 필요한 이유다.

---

## 3. OpenCV 실험 노트북들

노트북 16개 중 **13개가 OpenCV 실험**이고, 그중 다수가 사실상 같은 작업
(컨투어로 물체 영역을 찾아 크롭)의 반복이다.

| # | 문제 | 확인 |
|---|---|---|
| 1 | `contours_xy = np.array(contours)` | 컨투어마다 점 개수가 달라 **NumPy 1.24+ 에서 `ValueError`** |
| 2 | 이중 for 로 x·y min/max 탐색 | 반복마다 리스트 전체를 다시 훑음 (O(n²)) → `cv2.boundingRect(np.vstack(...))` 한 줄 |
| 3 | `cv2.imshow` + `waitKey` | 노트북에서 **커널이 멈춘 것처럼 보인다** → 결과 이미지가 하나도 안 남았다 |
| 4 | `cv2.imread('Xc7y3Q_..._CV2.png')` | 저장소에 없는 파일 → `None` → 세 줄 뒤 `imshow` 에서 `size.width>0` 에러 |
| 5 | `img = cv.imread(...)` 인데 `cv2.warpAffine(...)` | 별칭이 섞여 `NameError` (`image 끝 기준으로 자르기.ipynb`) |
| 6 | `cv2.drawContours(blurb, ...)` | 정의되지 않은 변수 (`blurb` 는 주석 처리돼 있다, `Untitled3.ipynb`) |
| 7 | `cv2.destroyAllWIndows()` | 오타 (`cv2 test 1.ipynb`) |

3~7 은 **실행하지 않은 채 저장된 셀**이 그대로 남아 있다는 뜻이기도 하다.
`Restart & Run All` 을 한 번만 돌렸어도 전부 드러났을 오류들이다.

---

## 4. 노트북 관리 상태

| 항목 | 원본 | 지금 |
|---|---|---|
| `Untitled`~`Untitled6` | 7개 — 열기 전에는 내용을 알 수 없음 | 번호 + 주제 이름 |
| `.ipynb_checkpoints/` | 10개 커밋됨 (본체와 거의 중복). 그중 `이미지 마스킹-checkpoint.ipynb` 는 **본체가 없는** 삭제된 노트북의 잔해다 | `.gitignore` |
| `.idea/` | JetBrains 개인 설정 커밋 | `.gitignore` |
| `.DS_Store` | 6KB macOS 메타데이터 커밋 | `.gitignore` |
| 파이썬 버전 | 노트북마다 3.7.3 / 3.7.5 뒤섞임 | `requirements.txt` 로 명시 |
| 실행 상태 | 셀 번호가 뒤죽박죽, 출력 없는 셀 다수 | 전부 Restart & Run All |
| 데이터 | 개인 PC 경로 | 코드로 생성 |

자세한 기준은 [docs/notebook-hygiene.md](../docs/notebook-hygiene.md) 에 정리했다.
