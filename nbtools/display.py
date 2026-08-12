"""노트북 표시 헬퍼 — 한글 폰트와 OpenCV 이미지 인라인 출력."""

from __future__ import annotations

from pathlib import Path


def use_korean_font(size: int = 11) -> str | None:
    """matplotlib 한글 폰트를 설정하고 사용된 폰트 이름을 돌려준다.

    설정하지 않으면 한글이 전부 두부(□)로 나온다. matplotlib 기본 폰트(DejaVu Sans)에
    한글 글리프가 없기 때문이며, 경고만 뜨고 그림은 그려지므로 놓치기 쉽다.

    ``axes.unicode_minus=False`` 도 함께 끈다 — 유니코드 마이너스(−)가 한글 폰트에
    없어서 음수 축 라벨이 깨지는 별개의 문제다.
    """
    import matplotlib
    from matplotlib import font_manager

    candidates = ["AppleSDGothicNeo-Regular", "AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["font.size"] = size
            return name

    print("한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
    print("  macOS: 기본 설치됨 / Ubuntu: sudo apt install fonts-nanum / Windows: 맑은 고딕")
    return None


def imshow(img, title: str = "", size: float = 5.0, axis: bool = False):
    """OpenCV 이미지(BGR)를 노트북에 **인라인으로** 표시한다.

    ``cv2.imshow`` 는 별도 GUI 창을 띄우고 ``cv2.waitKey`` 로 블로킹한다. 주피터 커널
    안에서는 그 창의 이벤트 루프가 돌지 않아 **커널이 멈춘 것처럼 보인다** — 원본
    노트북들이 전부 밟은 문제다. 노트북에서는 matplotlib 로 그리는 것이 정답이다.

    OpenCV 는 BGR, matplotlib 는 RGB 이므로 채널 순서를 바꿔 준다.
    """
    import cv2
    import matplotlib.pyplot as plt

    if img is None:
        raise ValueError("이미지가 None 입니다 (cv2.imread 실패 여부를 먼저 확인하세요)")

    if img.ndim == 2:
        display, cmap = img, "gray"
    else:
        display, cmap = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None

    height, width = img.shape[:2]
    figure = plt.figure(figsize=(size, size * height / width))
    plt.imshow(display, cmap=cmap)
    if title:
        plt.title(title)
    if not axis:
        plt.axis("off")
    plt.tight_layout()
    return figure


def sample_image(path: str | Path | None = None):
    """실습용 이미지를 만든다 (외부 파일 없이 노트북이 돌아가도록)."""
    import cv2
    import numpy as np

    if path is not None and Path(path).exists():
        return cv2.imread(str(path))

    img = np.full((320, 480, 3), 245, np.uint8)
    cv2.rectangle(img, (40, 50), (180, 190), (200, 90, 20), -1)
    cv2.circle(img, (280, 120), 62, (60, 175, 60), -1)
    cv2.fillPoly(img, [np.array([[380, 50], [450, 190], [310, 190]])], (55, 55, 220))
    cv2.putText(img, "jupyterTest", (44, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
    return img
