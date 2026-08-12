# 설치와 문제 해결

## 1. 기본 설치

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nbtools          # 실습 데이터 생성 확인
jupyter lab
```

VS Code 나 PyCharm 에서 노트북을 열 거라면 `jupyterlab` 은 필요 없고 `ipykernel` 만 있으면 된다.

## 2. 가장 흔한 문제 — 커널과 가상환경 불일치

> `pip install pandas` 를 분명히 했는데 노트북에서 `ModuleNotFoundError: No module named 'pandas'`

터미널의 파이썬과 **노트북 커널의 파이썬이 다른 것**이다. 먼저 확인한다.

```python
import sys
print(sys.executable)     # 이 경로가 내 .venv 안인가?
```

### 해결 ① 가상환경을 커널로 등록

```bash
.venv/bin/python -m pip install ipykernel
.venv/bin/python -m ipykernel install --user --name jupytertest --display-name "Python (jupyterTest)"
```

주피터를 새로고침하고 우상단에서 `Python (jupyterTest)` 커널을 고른다.

### 해결 ② 노트북 안에서 현재 커널에 설치

```python
%pip install pandas       # 권장: 현재 커널의 파이썬에 설치된다
!pip install pandas       # 비권장: PATH 상의 다른 pip 일 수 있다
```

### 등록된 커널 목록 확인·정리

```bash
jupyter kernelspec list
jupyter kernelspec uninstall 오래된커널이름
```

## 3. 그래프의 한글이 □ 로 나온다

matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프가 없다. 경고만 뜨고 그림은 그려지므로
**깨진 채로 보고서에 들어가기 쉽다.**

```python
from nbtools import use_korean_font
use_korean_font()          # 폰트 자동 탐색 + axes.unicode_minus=False
```

직접 설정한다면 두 줄이다.

```python
import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"    # 아래 표 참고
matplotlib.rcParams["axes.unicode_minus"] = False     # 음수 기호(−) 깨짐 방지 (별개 문제다)
```

| OS | 폰트 이름 | 준비 |
|---|---|---|
| macOS | `AppleGothic`, `AppleSDGothicNeo-Regular` | 기본 설치됨 |
| Ubuntu | `NanumGothic` | `sudo apt install fonts-nanum` 후 `rm -rf ~/.cache/matplotlib` |
| Windows | `Malgun Gothic` | 기본 설치됨 |

설치된 한글 폰트 목록을 보려면:

```python
from matplotlib import font_manager
sorted({f.name for f in font_manager.fontManager.ttflist if "Gothic" in f.name or "Nanum" in f.name})
```

## 4. `cv2.imshow` 후 커널이 멈춘다

노트북에서는 `cv2.imshow` 를 쓰지 않는다. matplotlib 로 인라인 출력한다.
자세한 이유와 대안은 [08 노트북](../notebooks/08_opencv_in_jupyter.ipynb) 에 있다.

```python
from nbtools import imshow
imshow(img)                # BGR→RGB 변환까지 처리
```

이미 멈췄다면 **Kernel → Interrupt**, 안 되면 **Restart**.

## 5. 선택 설치 — Selenium

07 노트북은 브라우저 없이도 끝까지 읽히지만, 직접 실행해 보려면:

```bash
pip install selenium
```

Selenium 4.6 부터는 **chromedriver 를 따로 내려받지 않아도 된다**(Selenium Manager 가 처리).
2019년 자료에 나오는 "크롬 버전에 맞는 드라이버 받기" 절차는 이제 필요 없다.

## 6. 실습 데이터가 없다는 오류

```python
from nbtools import ensure_data
ensure_data()              # data/ 에 생성
```

터미널에서는 `python -m nbtools`. `data/` 는 `.gitignore` 대상이라 클론 직후에는 비어 있는 것이 정상이다.

## 7. 노트북이 열리지 않거나 깨졌다고 나올 때

`.ipynb` 는 JSON 이다. 편집 중 강제 종료 등으로 깨지면 열리지 않는다.

```bash
python -c "import json,sys; json.load(open(sys.argv[1])); print('JSON 정상')" notebooks/04_data_cleaning.ipynb
jupyter nbconvert --to notebook --inplace notebooks/04_data_cleaning.ipynb   # 정규화 시도
```

`.ipynb_checkpoints/` 안에 직전 저장본이 남아 있을 수 있다(이 저장소는 git 에서 제외했지만
로컬에는 계속 생성된다).
