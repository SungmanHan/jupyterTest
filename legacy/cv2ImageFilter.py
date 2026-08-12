import cv2,sys
import numpy as np

point_list = []
count = 0

def rename(pathArg):
    orgPath = pathArg
    tempSplit = orgPath.split("/")
    orgFileName = tempSplit[-1]
    tempSplit = orgFileName.split(".")
    newFileName = tempSplit[0] +"_CV2."+ tempSplit[1]
    return orgPath.replace(orgFileName,newFileName)


def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)

def image_warp(imgPath):
    cv2.namedWindow('original')
    cv2.setMouseCallback('original', mouse_callback)
    img_original = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    while (True):

        cv2.imshow("original", img_original)

        height, weight = img_original.shape[:2]

        if cv2.waitKey(1) & 0xFF == 32:  # spacebar를 누르면 루프에서 빠져나옵니다.
            break

    pts1 = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
    pts2 = np.float32([[0, 0], [weight, 0], [0, height], [weight, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_result = cv2.warpPerspective(img_original, M, (weight, height))

    cv2.imwrite(imgPath, img_result, [int(cv2.IMWRITE_PNG_COMPRESSION), 100])

    return imgPath

def cv2ImageFilter(pathArg):
    #imgPath = image_warp(pathArg)
    imgPath = pathArg
    rImgPath = rename(imgPath)

    img = cv2.imread(imgPath,cv2.IMREAD_COLOR)

    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    img = cv2.filter2D(img, -1, kernel_sharpen_3)

    img = img + (-32,-32,-32)
    image_gray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=4)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(blur, 10, 200,apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    contours_xy = np.array(contours)
    contours_xy.shape

    # x의 min과 max 찾기
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    # image trim 하기
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    img_trim = img[y:y + h, x:x + w]

    cv2.imwrite(rImgPath, img_trim,[int(cv2.IMWRITE_PNG_COMPRESSION), 100])
    
    return rImgPath

#print(cv2ImageFilter(sys.argv[1]))

print(cv2ImageFilter(sys.argv[1]))