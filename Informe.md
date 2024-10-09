<p align="center" width="100%">
    <img width="45%" src="https://aicogestion.org/wp-content/uploads/2018/06/Logo-Unr-1.png">
</p>

# Tecnicatura Universitaria en Inteligencia Artificial {align=center}
## FCEIA - UNR {align=center}
## Procesamiento de imágenes {align=center}


**Integrantes:**
- Giuliano Crenna (C-7438/1)
- Bruno Emmanuel Pace
- Mariano Sancho 

---
Reever

# **Problema 1:**
La ecualización del histograma global nos permite distribuir de forma uniforme los niveles de intensidad de la imagen en todo el rango disponible. Esto lo logramos con una transformación que maximiza el contrase de la imagen.

¿Como se logra resolver el problema?

**Procedimiento:**
Con la ecualización **global** del histograma buscamos mejorar el contraste de la imagen. La idea es redistribuír los niveles de intensidad de la imagen. Esto lo logramos calculando el histograma de de la imagen.
La ecualización **local** del histograma sigue la misma lógica que la **global** pero se centra en la transformación de un pixel y sus vecinos cercanos, Esto mejora el contraste en áreas específicas de a imagen.

1. Definimos una ventana local de tamaño $M \times N$ por ejemplo de $15 \times 15$. Esta ventana define la region sobre la cual se aplica la ecualización.
2. Se calcula el histograma que cuenta la cantidad de pixeles cara cada nivel de intensidad $h(i)$.
3. Se calcula la función de distribución acumulativa **(CDF)**, que es la suma acumulada del histograma normalizado.
$$
CDF(i) = \sum_{j=0}^{i} \frac{h(j)}{N}\\
N=\text{ Cantidad total de pixeles.}
$$
4. Ahora mapeamos todos los niveles de intensidad $i$ utilizando $CDF(i)$

$$
i'=round(CDF(i)*(L-1))\\
L = \text{ Cantidad de niveles de intensidad, 256 para 8 bits, ya que incluye al 0.}
$$
5. Desplazamos la ventana pizel a pixel por toda la imagen.
