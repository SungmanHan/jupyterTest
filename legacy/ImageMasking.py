import numpy as np
import cv2,sys

def ImageMak(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.rectangle(img, (508, 178), (647, 215), (0, 0, 0), -1)
    cv2.rectangle(img, (770, 390), (860, 417), (0, 0, 0), -1)
    cv2.rectangle(img, (350, 486), (510, 520), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Verified ID', (45, 530), font, 1, (0, 0, 255), 2)
    cv2.imwrite(path, img,[int(cv2.IMWRITE_PNG_COMPRESSION), 100])
    return "success"


#ImageMak('C:/Users/gridone/Desktop/imagesdata/VQbuP1_2019114_13546_28_CV2.png')
print(ImageMak(sys.argv[1]))