# Tecnicatura Universitaria en Inteligencia Artificial {align=center}
## FCEIA - UNR {align=center}
## Procesamiento de imágenes {align=center}

**Integrantes:**
- Giuliano Crenna (C-7438/1)
- Bruno Emmanuel Pace
- Mariano Sancho 

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

python3 main.py
```

### Windows
```bash
.\.venv\Scripts\activate

python main.py
```