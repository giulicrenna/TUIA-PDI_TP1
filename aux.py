# --- Hough Lineas --------------------------------------------------------------------------------
# Tutorial: 
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#   https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
f = cv2.imread('contornos.png')             # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
imshow(gray)

edges = cv2.Canny(gray, 100, 170, apertureSize=3)
cv2.imshow('imagen', edges)

f_lines = f.copy()
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=170)  
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b


img_cols = np.sum(img, axis=0)
img_rows = np.sum(img, axis=1)

print(f'Tenemos {len(img_cols)} columnas y {len(img_rows)} filas')

