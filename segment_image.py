import cv2
import numpy as np

def segment_image(image_path, output_path):
    image_path = "images/1.jpeg"
    print(f"Görüntü okunuyor: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Hata: Görüntü okunamadı!")
        return

    # 1. Ön İşleme (Preprocessing)
    # Gürültüyü azaltmak için Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Kenar Tespiti (Edge Detection)
    # Canny algoritması ile kenarları bul
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Morfolojik İşlemler (Morphology)
    # Kenarları birleştirmek ve boşlukları doldurmak için
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilation: Kenarları kalınlaştır
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Closing: Küçük delikleri kapat
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Kontur Bulma (Segmentation)
    # Kapalı alanları (nesneleri) bul
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Görselleştirme
    # Orijinal görüntü üzerine segmentleri çiz
    result = image.copy()
    
    # Rastgele renklerle segmentleri boya
    for i, contour in enumerate(contours):
        # Çok küçük alanları yoksay (Gürültü)
        if cv2.contourArea(contour) < 500:
            continue
            
        color = np.random.randint(0, 255, size=(3,)).tolist()
        
        # Kontur içini doldur (Maskeleme)
        cv2.drawContours(result, [contour], -1, color, 2) # Sınırları çiz
        
        # Yarı saydam dolgu
        overlay = result.copy()
        cv2.drawContours(overlay, [contour], -1, color, -1)
        cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)

    # Sonuçları kaydet
    cv2.imwrite(output_path, result)
    cv2.imwrite("edges_1.jpeg", edges)
    cv2.imwrite("mask_1.jpeg", closed)
    
    print(f"İşlem tamamlandı.")
    print(f"Sonuç kaydedildi: {output_path}")
    print(f"Kenar haritası: edges_1.jpeg")
    print(f"Maske: mask_1.jpeg")

if __name__ == "__main__":
    segment_image("1.jpeg", "segmented_1.jpeg")
