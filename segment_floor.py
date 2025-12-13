import cv2
import numpy as np

def segment_floor(image_path, output_path):
    print(f"Görüntü okunuyor: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Hata: Görüntü okunamadı!")
        return

    h, w = image.shape[:2]

    # 1. Ön İşleme
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    
    # 2. Kenar Tespiti
    # Zemin genellikle dokusuzdur, nesneler kenar oluşturur.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100) # Hassas eşikler
    
    # 3. Kenarları Kapatma (Morphology)
    # Kenarların arasındaki boşlukları doldur ki "su sızmasın"
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    
    # 4. Flood Fill (Su Doldurma) Algoritması
    # Mantık: Ekranın alt-orta noktasından (ayak ucumuz) su dökmeye başla.
    # Su, kenarlara (duvarlara/engellere) çarpana kadar yayılacaktır.
    
    # Kenar haritasını ters çevir (0: Siyah/Engel, 255: Beyaz/Boşluk)
    # Ancak floodFill için maske kullanacağız.
    
    # FloodFill için maske (Görüntüden 2 piksel büyük olmalı)
    # Maske: 0 = Doldurulabilir, 1 = Engel
    h, w = edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Kenarları maskeye kopyala
    # dilated_edges'de 255 olan yerler (kenarlar) maskede 1 olmalı (engel)
    # dilated_edges'de 0 olan yerler (boşluk) maskede 0 olmalı (doldurulabilir)
    
    # Maskeyi hazırla: Kenar olan yerleri 1 yap
    mask_content = dilated_edges.copy()
    # Kenarları kalınlaştırılmış görüntüden maskeye aktar
    mask[1:-1, 1:-1] = mask_content // 255 # 255 -> 1
    
    # Seed Point (Başlangıç Noktası): Ekranın alt ortası
    seed_point = (w // 2, h - 10)
    
    # Eğer başlangıç noktası zaten bir kenarın üzerindeyse, biraz yukarı/sağa/sola kaydırıp dene
    if mask[seed_point[1]+1, seed_point[0]+1] == 1:
        print("Uyarı: Başlangıç noktası engel üzerinde, alternatif aranıyor...")
        found = False
        for y in range(h-10, h-100, -5):
            for x in range(w//2 - 50, w//2 + 50, 5):
                if mask[y+1, x+1] == 0:
                    seed_point = (x, y)
                    found = True
                    break
            if found: break
    
    # Flood Fill uygula
    # Boş bir tuval oluştur (Siyah)
    floor_map = np.zeros((h, w), np.uint8)
    
    # FloodFill doğrudan floor_map'i boyamaz, maskeyi günceller veya görüntüyü boyar.
    # Biz geçici bir görüntü üzerinde yapalım.
    temp_img = np.zeros((h, w), np.uint8)
    
    # 255 rengiyle doldur
    cv2.floodFill(temp_img, mask, seed_point, 255, flags=4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
    
    # Maske artık doldu. Maskenin '1' olduğu yeni yerler zemin.
    # Ancak floodFill temp_img'yi boyadı.
    floor_mask = temp_img
    
    # 5. Görselleştirme
    result = image.copy()
    
    # Zemini Yeşil ile boya (Yarı saydam)
    overlay = result.copy()
    overlay[floor_mask == 255] = [0, 255, 0] # BGR: Yeşil
    
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
    
    # Sınırları çiz (Zemin ile engel arasındaki çizgi)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2) # Kırmızı sınır
    
    # Kaydet
    cv2.imwrite(output_path, result)
    cv2.imwrite("floor_mask.jpeg", floor_mask)
    cv2.imwrite("edges_debug.jpeg", dilated_edges)
    
    print(f"Zemin tespiti tamamlandı.")
    print(f"Sonuç: {output_path}")

if __name__ == "__main__":
    segment_floor("images/1.jpeg", "floor_segmented.jpeg")
