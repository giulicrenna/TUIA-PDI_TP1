import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

PATH_IMAGENES: str = os.path.join(os.getcwd(), "data")

Matlike = np.ndarray

def cargar_y_procesar_imagen(filepath):
    """
    Función que carga y umbraliza la imagen.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    return img_bin

img = cargar_y_procesar_imagen(os.path.join(PATH_IMAGENES, 'examen_2.png'))

cv2.imshow('imagen umbralizada', img)
cv2.waitKey(0)

edges = cv2.Canny(img, 100, 170, apertureSize=5)
cv2.imshow('imagen', edges)
cv2.waitKey(0)

f_lines = img.copy()
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=180)
print(f'Lineas encontradas: {len(lines)}')

for i in range(0,len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a ))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(f_lines,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imshow('Lineas', f_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()