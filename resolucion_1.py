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

def ecualizacion_local_histograma(img: Matlike, kernel_size: int) -> Matlike:
    border_size: int = kernel_size // 2
    img_con_padding: Matlike = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)
    
    salida: Matlike = np.zeros_like(img)

    for i in range(border_size, img_con_padding.shape[0] - border_size):
        for j in range(border_size, img_con_padding.shape[1] - border_size):
            ventana: Matlike = img_con_padding[i - border_size:i + border_size + 1, j - border_size:j + border_size + 1]
            
            ventana_ecualizada: Matlike = cv2.equalizeHist(ventana)
            
            salida[i - border_size, j - border_size] = ventana_ecualizada[border_size, border_size]

    return salida
    


def problema_1(path: str) -> None:
    img: Matlike = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    ventana_size: int = 20
    img_ecualizada: Matlike = ecualizacion_local_histograma(img=img,
                                                        kernel_size=ventana_size)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Imagen con Ecualización Local (Ventana {ventana_size}x{ventana_size})')
    plt.imshow(img_ecualizada, cmap='gray')
    plt.axis('off')

    plt.show()
    
if __name__ == '__main__':
    problema_1(path=os.path.join(PATH_IMAGENES, 'Imagen_con_detalles_escondidos.tif'))
    