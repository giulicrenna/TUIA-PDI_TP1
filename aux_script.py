import cv2
import numpy as np

img = cv2.imread('data/examen_5.png', cv2.IMREAD_GRAYSCALE)

_, img_th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

img_cols = np.sum(img_th, axis=0)
img_rows = np.sum(img_th, axis=1)

th_col = 50 #500
th_row = 50 #3500

img_cols_th = img_cols > th_col
img_rows_th = img_rows > th_row

vertical_lines = np.where(img_cols_th)[0]
horizontal_lines = np.where(img_rows_th)[0]

def find_line_bounds(lines):
    line_bounds = []
    start = None
    for i, val in enumerate(lines):
        if val and start is None:
            start = i
        elif not val and start is not None:
            line_bounds.append((start, i-1))
            start = None
    if start is not None:
        line_bounds.append((start, len(lines)-1))
    return line_bounds

vertical_bounds = find_line_bounds(img_cols_th)
horizontal_bounds = find_line_bounds(img_rows_th)


img_lines = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
for (x1, x2) in vertical_bounds:
    cv2.line(img_lines, (x1, 0), (x1, img.shape[0]), (0, 255, 0), 1)
for (y1, y2) in horizontal_bounds:
    cv2.line(img_lines, (0, y1), (img.shape[1], y1), (0, 255, 0), 1)

cv2.imshow('Líneas detectadas', img_lines)
cv2.waitKey(0)

for i in range(len(horizontal_bounds)-1):
    for j in range(len(vertical_bounds)-1):
        celda_img = img_th[horizontal_bounds[i][0]:horizontal_bounds[i+1][1],
                           vertical_bounds[j][0]:vertical_bounds[j+1][1]]
        

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_img, 8, cv2.CV_32S)

        th_area = 4000#3800  
        ix_area = stats[:, -1] > th_area
        stats_filtered = stats[ix_area, :]
        
        for stat in stats_filtered:
            x, y, w, h, area = stat
            componente = celda_img[y:y+h, x:x+w]

            #cv2.imshow('Componente detectada', componente)
            #cv2.waitKey(0)
            #cv2.imshow('celda',celda_img)
            #cv2.waitKey(0)


cv2.destroyAllWindows()


