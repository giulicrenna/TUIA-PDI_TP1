import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

PATH_IMAGENES: str = os.path.join(os.getcwd(), 'data')

clave_respuestas: Dict[int, str] = {1: 'C', 2: 'B', 3: 'A', 4: 'D', 5: 'B', 6: 'B', 7: 'A', 8: 'B', 9: 'D', 10: 'D'}
tamaño_ventana: Tuple[int, int] = (50, 50)
umbral_marcado: int = 200

Matlike = np.ndarray

def ecualizacion_local_histograma(img: Matlike, kernel_size: Tuple) -> Matlike:
    m, n = kernel_size[0], kernel_size[1]
    img_con_padding: Matlike = cv2.copyMakeBorder(img, n, n, m, m, cv2.BORDER_REPLICATE)
    
    salida: Matlike = np.zeros_like(img)

    for i in range(n, img_con_padding.shape[0] - n):
        for j in range(m, img_con_padding.shape[1] - m):
            ventana: Matlike = img_con_padding[i - n:i + n + 1, j - m:j + m + 1]
            
            ventana_ecualizada: Matlike = cv2.equalizeHist(ventana)
            
            salida[i - n, j - m] = ventana_ecualizada[m, n]

    return salida
    


def problema_1(path: str, window_size: Tuple) -> None:
    window_size_x, window_size_y = window_size
    img: Matlike = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img_ecualizada: Matlike = ecualizacion_local_histograma(img=img,
                                                        kernel_size=window_size)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Imagen con Ecualización Local (Ventana {window_size_x}x{window_size_y})')
    plt.imshow(img_ecualizada, cmap='gray')
    plt.axis('off')

    plt.show()
    
if __name__ == '__main__':
    problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'),window_size=(10,10))
    