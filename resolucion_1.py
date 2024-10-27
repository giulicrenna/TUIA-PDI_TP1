import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

PATH_IMAGENES: str = os.path.join(os.getcwd(), 'data')

Matlike = np.ndarray
def ecualizacion_local_histograma(img: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """
    Se solucionaron los problemas.
    1. Compatibilidad con ventanas M x N cualesquiera.
    2. Reducción del ruido en la imagen de salida.
    """
    m, n = kernel_size
    """
    Esto garantiza que, sin importar el tamaño de la ventana,
     siempre haya suficiente espacio alrededor de cada píxel para que la ventana se centre correctamente en él.
    """
    pad_y, pad_x = n // 2, m // 2

    img = cv2.GaussianBlur(img, (5, 5), 0)

    img_con_padding = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    
    salida = np.zeros_like(img)
    
    for i in range(pad_y, img_con_padding.shape[0] - pad_y):
        for j in range(pad_x, img_con_padding.shape[1] - pad_x):
            ventana = img_con_padding[i - pad_y:i + pad_y + 1, j - pad_x:j + pad_x + 1]
            
            ventana_ecualizada = cv2.equalizeHist(ventana)
            
            salida[i - pad_y, j - pad_x] = ventana_ecualizada[pad_y, pad_x]

    return salida
    
def problema_1(path: str, window_size: Tuple) -> None:
    window_size_x, window_size_y = window_size
    img: Matlike = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img_ecualizada: Matlike = ecualizacion_local_histograma(img=img,
                                                        kernel_size=window_size)

    plt.figure(figsize=(12, 8))  

    plt.subplot(2, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f'Imagen con Ecualización Local (Ventana {window_size_x}x{window_size_y})')
    plt.imshow(img_ecualizada, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Histograma Imagen Original')
    plt.hist(img.flatten(), 256, [0, 256], color='blue')
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.title('Histograma Imagen Ecualizada')
    plt.hist(img_ecualizada.flatten(), 256, [0, 256], color='blue')
    plt.xlim([0, 256])

    plt.tight_layout()  
    plt.show()


"""
Visualizaciones realizadas.
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(3,3))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(4,5))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(7,5))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(10,10))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(10,12))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(12,10))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(12,15))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(20,20))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(100,100))
problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(5,10))
"""


if __name__ == '__main__':
    problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(10,10))


    
