import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


PATH_IMAGENES: str = os.path.join(os.getcwd(), "data")
Matlike = np.ndarray


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    """
    Imprime imágenes, con variables adecuadas.
    """
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


def cargar_y_procesar_imagen(filepath: str) -> Matlike:
    """
    Función que carga y umbraliza la imagen.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    return img_bin



####################################################
def detectar_lineas(img: Matlike) -> List[Tuple[int,int,int]]:
    """
    Detecta líneas horizontales utilizando morfología OPEN. Retorna los renglones ordenados.
    """
    
    lineas: Matlike = np.zeros_like(img)

    kernel_horizontal: Matlike = np.ones((1, 40), np.uint8)

    lineas_horizontales = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_horizontal)

    contornos, _ = cv2.findContours(
        lineas_horizontales, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_renglon = max(cv2.boundingRect(contorno)[2] for contorno in contornos)
    renglones: List = []
    

    for contorno in contornos:
        x, y, w, _ = cv2.boundingRect(contorno)

        # Guarda solo las lineas cuya longitud sea menor a la maxima encontrada (con una tolerancia de 10 pixeles)
        if w < max_renglon-10:
            #cv2.line(lineas, (x, y), (x + w, y), (255, 0, 0), 2)
            renglones.append((x,y,w))

        # Se ordenan los renglones de forma ascendiente la coordenada 'y'. En caso de que haya dos valores iguales, 
        # se ordena de forma ascendiente la coordenada 'x'.  
        renglones_sorted = sorted(renglones, key=lambda renglon: (renglon[1], renglon[0])) 
    
        #print(renglones_sorted)
    #imshow(lineas)
    return renglones_sorted

    
def check_and_set(renglones: List[Tuple]) -> Dict[str, Tuple[int, int, int]]:
    """
    Itera sobre los renglones, clasificando en: Nombre, Fecha, Clase y Preguntas de la 1 a la 10.
    """

    mapa_subconjuntos: Dict[str, Tuple[int, int, int]] = {
        "Nombre": (),
        "Date": (),
        "Class": (),
        "Pregunta 1": (),
        "Pregunta 6": (),
        "Pregunta 2": (),
        "Pregunta 7": (),
        "Pregunta 8": (),
        "Pregunta 3": (),
        "Pregunta 9": (),
        "Pregunta 4": (),
        "Pregunta 5": (),
        "Pregunta 10": (),
        }

    mapa_subconjuntos["Nombre"] = renglones[0]  
    mapa_subconjuntos["Date"] = renglones[1]    
    mapa_subconjuntos["Class"] = renglones[2]   
      
    for i, key in enumerate(mapa_subconjuntos.keys()):
        mapa_subconjuntos[key] = renglones[i]

    return mapa_subconjuntos


def obtener_respuesta(img: Matlike, mapa_subconjuntos: Dict[str, Tuple[int, int, int]]): 
    """
    Itera sobre mapa_subconjuntos para obtener las respuestas croppeadas de la imagen. 
    """
    respuestas = {}
    
    for key, (x, y, w) in mapa_subconjuntos.items():
        
        # Definir la altura del recorte según la clave
        if key in ["Nombre", "Date", "Class"]:
            h = 20  # Altura de 20 píxeles para nombre, date y class
        else:
            h = 13  # Altura de 15 píxeles para las demás claves
        
        respuesta = img[y - h: y, x:x + w]

        # Almacenar el recorte en el diccionario de respuestas
        respuestas[key] = respuesta
    
    
    # for key, recorte in respuestas.items(): 
    #     imshow(recorte, title = key)
        
    return respuestas


####################################################
def validaciones_header(key, recorte): 

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(recorte, connectivity=8)

    im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
    # for centroid in centroids:
    #     cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
    # for st in stats:
        # cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
    # imshow(img=im_color, color_img=True)
    
    if key == 'Nombre': 
        
        distancia_minima = 12
        
        # Cant. de letras
        if 25 >= num_labels >= 3:
        
            # Se verifica la existencia de un espacio entre "Nombre" y "Apellido"
            x_ant = stats[1][0]
            
            for i in range(2, num_labels):
                x, _, _, _, _ = stats[i]
                
                if x - x_ant > distancia_minima:
                    return True
                
                x_ant = x
        
        return False
    
    if key == 'Date':
        if num_labels == 9: # el rectangulo del fondo suma un objeto
            return True        
        return False   
    
    if key == 'Class':
        if num_labels == 2: # el rectangulo del fondo suma un objeto
            return True        
        return False


####################################################
def contar_pixeles(
    img_array: Matlike, umbral_min: int = 100, umbral_max: int = 200
) -> int:
    """
    Esta función recibe una imagen en forma de array de NumPy y retorna la cantidad de píxeles
    que forman parte de los bordes, contando la cantidad de pixeles blancos.
    """
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array

    cantidad_pixeles = np.count_nonzero(img_gray)

    return cantidad_pixeles    


####################################################
def validaciones_preguntas(key: str, recorte: Matlike) -> bool: 
    """
    Corrige cada ejercicio. Retorna True o False
    """
    
    cant_pixeles: Dict[str, int] = {"A": 28, "B": 33, "C": 22, "D": 29}

    mapa_correctas: Dict[str, Dict[str, Tuple[int, int]]] = {
        "Pregunta 1": "C",
        "Pregunta 2": "B",
        "Pregunta 3": "A",
        "Pregunta 4": "D",
        "Pregunta 5": "B",
        "Pregunta 6": "B",
        "Pregunta 7": "A",
        "Pregunta 8": "B",
        "Pregunta 9": "D",
        "Pregunta 10": "D",
        }
   
    respuesta_alumno = contar_pixeles(recorte)
    respuesta_correcta = cant_pixeles[mapa_correctas[key]]

    if respuesta_alumno == respuesta_correcta :
        return True
    else:
        return False
    


####################################################
# resolucion #
####################################################


 
def definir_examen(img: Matlike):
    renglones = detectar_lineas(img)
    #print(renglones)
    mapa_subconjuntos = check_and_set(renglones)
    #print(mapa_subconjuntos)
    respuestas = obtener_respuesta(img, mapa_subconjuntos)
    #print(respuestas)
    correctas: int = 0
    examen_final = {}
    for key, recorte in respuestas.items():
        
        if key in ["Nombre", "Date", "Class"]:
            if validaciones_header(key, recorte):
                examen_final[key] = 'OK'
                print(f'{key}: OK')
            else: 
                examen_final[key] = 'MAL'
                print(f'{key}: MAL')
            continue

        if validaciones_preguntas(key, recorte):
            examen_final[key] = 'Correcto'
            correctas += 1
        else: 
            examen_final[key] = 'Incorrecto'
    
    examen_final['Nota'] = correctas

    if correctas >= 6:
        examen_final['Condicion'] = 'Aprobado'
    else:
        examen_final['Condicion'] = 'Reprobado'

    return examen_final, respuestas['Nombre']
     
            
def imprimir_lista_final(aprobados:List, reprobados: List) -> None:
    total_plots = len(aprobados) + len(reprobados)

    fig, axs = plt.subplots(nrows=1, ncols=total_plots, figsize=(15, 5))

    for i, arr in enumerate(aprobados):
        axs[i].imshow(arr, cmap='viridis')  
        axs[i].set_title('Aprobado {}'.format(i + 1))
        axs[i].axis('off') 

    for j, arr in enumerate(reprobados):
        axs[len(aprobados) + j].imshow(arr, cmap='gray')  
        axs[len(aprobados) + j].set_title('Reprobado {}'.format(j + 1))
        axs[len(aprobados) + j].axis('off') 

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    aprobados: List = []
    reprobados: List = []

    for i in range(1, 6):
        relative_path: str = f'examen_{i}.png'
        
        print(f'Examen {i}: \n')
        
        path: str = f'examen_{i}.png'
        img = cargar_y_procesar_imagen(os.path.join(PATH_IMAGENES, path))
        
        examen_final, nombre = definir_examen(img)
        

        if examen_final['Nombre'] == 'MAL' or examen_final['Date'] == 'MAL' or examen_final['Class'] == 'MAL':
            print(f'Examen {i} inválido.')

        if examen_final['Condicion'] == 'Aprobado':
            aprobados.append(nombre)
        else:
            reprobados.append(nombre)
        for i in range(1,11):
            print(f'Pregunta {i} :',examen_final[f'Pregunta {i}'])
                        
        print('------------------------------------')
        
    imprimir_lista_final(aprobados, reprobados)

            
