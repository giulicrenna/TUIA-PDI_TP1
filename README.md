# Tecnicatura Universitaria en Inteligencia Artificial 
## FCEIA - UNR 
## Procesamiento de imágenes 


**Integrantes:**
- Crenna, Giuliano. Legajo: C-7438/1.
- Pace, Bruno. Legajo: P-5295/7.
- Sancho Almenar, Mariano. Legajo: S-5778/9.

---

## Preparación del Entorno

### Linux
```bash
sudo apt install tesseract-ocr

python3 -m venv .venv

source .venv/bin/activate

pip3 install -r requirements.txt
```

### Windows
Primero intallar Tesseract desde el siguiente link: [Tesseract](https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe)
```bash
python -m venv .venv

pip install -r requirements.txt

.\.venv\Scripts\activate
```

## Ejecución

### Linux
```bash
source .venv/bin/activate

python3 resolucion_1.py
python3 resolucion_2.py
```

### Windows
```bash
.\.venv\Scripts\activate


python resolucion_1.py
python resolucion_2.py
```
