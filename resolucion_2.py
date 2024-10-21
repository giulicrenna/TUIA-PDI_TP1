import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

PATH_IMAGENES: str = os.path.join(os.getcwd(), "data")

row_height: int = 15

Matlike = np.ndarray

mapa_subconjuntos: Dict[str, Dict[str, Tuple[int, int]]] = {
    "nombre": (),
    "date": (),
    "class": (),
    "1": (),
    "2": (),
    "3": (),
    "4": (),
    "5": (),
    "6": (),
    "7": (),
    "8": (),
    "9": (),
    "10": (),
}

mapa_correctas: Dict[str, Dict[str, Tuple[int, int]]] = {
    "1": "C",
    "2": "B",
    "3": "A",
    "4": "D",
    "5": "B",
    "6": "B",
    "7": "A",
    "8": "B",
    "9": "D",
    "10": "D",
}

cant_bordes: Dict[str, int] = {"A": 27, "B": 33, "C": 22, "D": 29}

validaciones = {
    "nombre": lambda cant_palabras: cant_palabras <= 25,
    "date": lambda cant_palabras: cant_palabras == 8,
    "class": lambda cant_palabras: cant_palabras == 1,
}


def cargar_y_procesar_imagen(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    return img_bin


def check_and_set(coord: Tuple[int, int, int, int]) -> str:
    x, y, w, h = coord

    if y < 45:
        if x < 254:
            return "nombre"
        elif x > 290 and x < 371:
            return "date"
        else:
            return "class"

    if x < 270:
        if y > 60 and y < 90:
            return "1"
        elif y > 170 and y < 215:
            return "2"
        elif y > 310 and y < 350:
            return "3"
        elif y > 410 and y < 480:
            return "4"
        elif y > 547 and y < 615:
            return "5"
    else:
        if y > 60 and y < 90:
            return "6"
        elif y > 170 and y < 215:
            return "7"
        elif y > 310 and y < 350:
            return "8"
        elif y > 420 and y < 500:
            return "9"
        elif y > 547 and y < 615:
            return "10"


def detectar_lineas_horizontales(img_bin: np.ndarray) -> None:
    """
    Detecta líneas horizontales en una imagen y retorna las coordenadas de las mismas.
    """
    global mapa_subconjuntos

    kernel_horizontal = np.ones((1, 40), np.uint8)

    lineas_horizontales = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_horizontal)

    contornos, _ = cv2.findContours(
        lineas_horizontales, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        # Verifico si es encabezado y establezco el tamaño de los renglones
        y = y - 19 if y <= 45 else y - 14
        h = 17 if y <= 45 else 14

        # Verifico la anchura de la linea horizontal
        if w < 200:
            # cv2.rectangle(img_bin, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Verifico y agrego al mapa de caracteres
            key = check_and_set((x, y, x + w, y + h))

            mapa_subconjuntos[key] = (x, y, x + w, y + h)


def contar_letras(chunk: np.ndarray) -> int:
    count: int = 0
    is_col: bool = True
    _, num_columnas = chunk.shape

    for col in range(num_columnas):
        columna = chunk[:, col]

        if np.count_nonzero(columna == 255) == 0:
            if not is_col:
                count += 1

            is_col = True
        else:
            is_col = False

    return count


def obtener_recortes(
    img: np.ndarray, mapa_subconjuntos: Dict[str, Tuple[int, int, int, int]]
) -> Dict[str, Any]:
    """
    Esta función recibe una imagen y un mapa de bounding boxes en formato (x, y, x + w, y + h),
    y devuelve un diccionario con los recortes de las regiones de interés.
    """

    recortes = {}

    for key, (x1, y1, x2, y2) in mapa_subconjuntos.items():
        recorte = img[y1:y2, x1:x2]

        recortes[key] = recorte

    return recortes


def contar_pixeles(
    img_array: Matlike, umbral_min: int = 100, umbral_max: int = 200
) -> int:
    """
    Esta función recibe una imagen en forma de array de NumPy y retorna la cantidad de píxeles
    que forman parte de los bordes, utilizando el detector de bordes Canny de OpenCV.
    """
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array

    cantidad_bordes = np.count_nonzero(img_gray)

    return cantidad_bordes


def determinar_letra(cant_pixeles: int, mapa_bordes: Dict = cant_bordes) -> str:
    for item in mapa_bordes.items():
        letra, cant = item

        if abs(cant - cant_pixeles) <= 1:
            return letra

    return "X"


def plot_recortes(recortes) -> None:
    for item in recortes.items():
        plt.figure(figsize=(10, 5))
        plt.title(f"Imagen: {item[0]}")
        plt.imshow(item[1], cmap="gray")
        plt.axis("off")

        plt.show()

def plot_recorte_unico(recorte) -> None:
    plt.figure(figsize=(10, 5))
    plt.title(f"Nombre")
    plt.imshow(recorte, cmap="gray")
    plt.axis("off")

    plt.show()


def definir_nota(filepath: str):
    img: str = os.path.join(PATH_IMAGENES, filepath)

    img_procesada = cargar_y_procesar_imagen(filepath=img)

    detectar_lineas_horizontales(img_bin=img_procesada)

    recortes = obtener_recortes(img=img_procesada, mapa_subconjuntos=mapa_subconjuntos)
    cant_correctas: int = 0

    for key, recorte in recortes.items():
        cant_palabras: int = contar_letras(recorte)

        if key in ["date", "nombre", "class"]:
            if validaciones[key](cant_palabras):
                print(f"{key.capitalize()}: OK")
            else:
                print(f"{key.capitalize()}: MAL")

            continue

        borde: int = contar_pixeles(recorte)
        letra: str = determinar_letra(borde)
        
        if letra == mapa_correctas[key]:
            print(f"Pregunta {key}: OK")
            cant_correctas = cant_correctas+1
        else: print(f"Pregunta {key}: MAL")

    return recortes['nombre'], cant_correctas
    
#plot_recorte_unico(recortes["nombre"])

def imprimir_lista_final(aprobados:List, reprobados: List) -> None:
    total_plots = len(aprobados) + len(reprobados)

    fig, axs = plt.subplots(nrows=1, ncols=total_plots, figsize=(15, 5))

    for i, arr in enumerate(aprobados):
        axs[i].imshow(arr, cmap='viridis')  
        axs[i].set_title('Aprobado {}'.format(i + 1))
        axs[i].axis('off') 

    for j, arr in enumerate(reprobados):
        axs[len(aprobados) + j].imshow(arr, cmap='plasma')  
        axs[len(aprobados) + j].set_title('Reprobado {}'.format(j + 1))
        axs[len(aprobados) + j].axis('off') 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    examenes: Dict[int, Tuple[np.ndarray, int]] = {}

    for i in range(1, 6):
        relative_path: str = f'examen_{i}.png'
        examenes[i] = definir_nota(relative_path)
    
    aprobados: List[np.ndarray] = []
    reprobados: List[np.ndarray] = []

    for value in examenes.values():
        if value[1] >= 6:
            aprobados.append(value[0])
        else: 
            reprobados.append(value[0])

    imprimir_lista_final(aprobados,reprobados)
    
