import cv2
import numpy as np

def leer_imagen(ruta):
    
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    return imagen

def corregir_perspectiva(imagen):
    
    # 1. Preprocesamiento
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(blur, 50, 150)
    
    # 2. Detección de contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
    
    # 3. Buscar contorno de documento (4 vértices)
    for cnt in contornos:
        perimetro = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimetro, True)
        if len(approx) == 4:
            doc_contorno = approx
            break
    else:
        return cv2.resize(imagen, (800,900))  # Si no encuentra contorno válido
    
    # 4. Ordenar puntos: [tl, tr, br, bl]
    puntos = doc_contorno.reshape(4,2)
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1)
    
    ordenados = np.array([
        puntos[np.argmin(suma)],       # Top-Left
        puntos[np.argmin(diferencia)], # Top-Right
        puntos[np.argmax(suma)],       # Bottom-Right
        puntos[np.argmax(diferencia)]  # Bottom-Left
    ], dtype=np.float32)
    
    # 5. Transformación de perspectiva
    destino = np.array([[0,0], [799,0], [799,899], [0,899]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordenados, destino)
    corregida = cv2.warpPerspective(imagen, M, (800,900))
    
    return corregida

def recortar_examen(imagen):
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Umbral adaptado con Otsu para binarizar la imagen
    _, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        # Si no se detecta ningún contorno, devuelve la imagen original
        return imagen
    
    # Se selecciona el contorno de mayor área
    contorno_grande = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_grande)
    recorte = imagen[y:y+h, x:x+w]
    return recorte


def main():
    
    # Procesar múltiples exámenes
    for i in range(1,13):
        img = leer_imagen(f"Examenes/examen{i}.jpg")
        img_corregida = corregir_perspectiva(img)
        img_recortada = recortar_examen(img_corregida)
        cv2.imwrite(f"Recortes/ recortado_{i}.jpg", img_recortada)
        

if __name__ == "__main__":
    main()