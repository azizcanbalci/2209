import cv2
import numpy as np
# from ultralytics import YOLO  # Artık VisionPipeline içinde
from vision_pipeline import VisionPipeline
import threading
from queue import Queue
import time
import os
from gtts import gTTS
import pygame

# Pygame mixer başlat
pygame.mixer.init()

# Global ses kuyruğu
speech_queue = Queue()
speech_thread_running = True

# Ses dosyaları için klasör
AUDIO_DIR = "audio_cache"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def create_audio_files():
    """Başlangıçta ses dosyalarını oluştur"""
    commands = {
        "SOL": "sola dön",
        "DÜZ": "düz git",
        "SAG": "sağa dön", 
        "DUR": "dur",
        "HAZIR": "Sistem hazır",
        "YAKIN": "Dikkat! Çok yakın engel",
        "UZAK": "Yol açık"
    }
    
    for key, text in commands.items():
        filepath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        if not os.path.exists(filepath):
            print(f"Ses dosyası oluşturuluyor: {text}")
            tts = gTTS(text=text, lang='tr')
            tts.save(filepath)
    print("Ses dosyaları hazır!")

def play_sound(command):
    """Ses dosyasını çal"""
    filepath = os.path.join(AUDIO_DIR, f"{command}.mp3")
    if os.path.exists(filepath):
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Ses çalma hatası: {e}")

def speech_worker():
    """
    Arka planda çalışan ses işçisi thread'i.
    Kuyruktan komutları alıp seslendirir.
    """
    while speech_thread_running:
        try:
            if not speech_queue.empty():
                komut = speech_queue.get(timeout=0.1)
                print(f"Seslendiriliyor: {komut}")
                play_sound(komut)
                speech_queue.task_done()
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Ses hatası: {e}")
            time.sleep(0.1)


def estimate_distance(y2: int, frame_height: int) -> tuple:
    """
    Nesnenin alt noktasına (y2) göre perspektif tabanlı mesafe tahmini yapar.
    
    Args:
        y2: Bounding box'ın alt y koordinatı (piksel)
        frame_height: Görüntü yüksekliği (piksel)
    
    Returns:
        tuple: (mesafe_metre, mesafe_kategori)
        mesafe_kategori: "YAKIN", "ORTA", "UZAK"
    """
    # Nesnenin alt noktasının frame'e oranı (0.0 - 1.0)
    # 1.0 = Ekranın en altı (Ayak ucu) -> Çok Yakın
    # 0.5 = Ekranın ortası (Ufuk çizgisi) -> Uzak
    ratio = y2 / frame_height
    
    # Perspektif tabanlı mesafe tahmini - REVIZE EDILDI (Daha dengeli)
    # Eşikler biraz düşürüldü (Çok katı olmaması için)
    if ratio > 0.85:  # Ayak ucunda (Ekranın alt %15'i) - 0.5m
        distance = 0.5
        category = "YAKIN"
    elif ratio > 0.70:  # 1-2 metre (Ekranın alt %30'u)
        distance = 1.0
        category = "YAKIN"
    elif ratio > 0.55:  # 3-4 metre (Ekranın alt yarısı)
        distance = 2.0
        category = "ORTA"
    elif ratio > 0.40:  # 5-6 metre
        distance = 4.0
        category = "ORTA"
    else:  # Ufuk çizgisine yakın veya üstünde
        distance = 8.0
        category = "UZAK"
    
    return distance, category


def get_closest_obstacle(obstacles: list, frame_height: int) -> tuple:
    """
    En yakın engeli ve mesafesini bulur.
    
    Args:
        obstacles: [(x1, y1, x2, y2), ...] formatında engel listesi
        frame_height: Görüntü yüksekliği
    
    Returns:
        tuple: (min_distance, category, closest_bbox) veya (None, None, None)
    """
    if not obstacles:
        return None, None, None
    
    min_distance = float('inf')
    closest_category = None
    closest_bbox = None
    
    for (x1, y1, x2, y2) in obstacles:
        # Yeni mesafe fonksiyonunu kullan (y2 ile)
        distance, category = estimate_distance(y2, frame_height)
        
        if distance < min_distance:
            min_distance = distance
            closest_category = category
            closest_bbox = (x1, y1, x2, y2)
    
    return min_distance, closest_category, closest_bbox


def speak_direction(direction: str, engine):
    """
    Yönü sesli olarak söyler.
    
    Args:
        direction: Söylenecek yön ("SOL", "DÜZ", "SAĞ", "DUR")
        engine: pyttsx3 engine nesnesi
    """
    if not engine:
        return
    
    # Türkçe telaffuz için dönüşüm
    turkish_pronunciations = {
        "SOL": "sola dön",
        "DÜZ": "düz git",
        "SAG": "sağa dön",
        "DUR": "dur"
    }
    
    text_to_speak = turkish_pronunciations.get(direction, direction)
    print(f"Söylenecek: {text_to_speak}")  # Debug için
    
    try:
        # Önceki konuşmayı durdur
        engine.stop()
        # Yeni komutu söyle
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"Ses hatasi: {e}")



def get_direction(obstacles: list, frame_width: int, frame_height: int, threshold: float = 0.3) -> str:
    """
    Engel konumlarına göre güvenli yönü hesaplar.
    
    Args:
        obstacles: Her engel için (x1, y1, x2, y2) formatında bounding box listesi
        frame_width: Görüntü genişliği
        frame_height: Görüntü yüksekliği
        threshold: Bölgenin "dolu" sayılması için minimum doluluk oranı (0-1)
    
    Returns:
        str: "DÜZ", "SOL", "SAG" veya "DUR"
    """
    # Görüntünün alt yarısını analiz edeceğiz (engeller yakınlık için daha önemli)
    bottom_half_start = frame_height // 2
    
    # 3 bölgenin sınırları
    left_end = frame_width // 3
    right_start = 2 * frame_width // 3
    
    # Her bölge için piksel bazlı doluluk hesapla
    region_height = frame_height - bottom_half_start
    region_areas = {
        "SOL": left_end * region_height,
        "DÜZ": (right_start - left_end) * region_height,
        "SAG": (frame_width - right_start) * region_height
    }
    
    # Her bölgedeki engel kapladığı alan
    region_obstacle_area = {
        "SOL": 0,
        "DÜZ": 0,
        "SAG": 0
    }
    
    for (x1, y1, x2, y2) in obstacles:
        # Sadece alt yarıdaki kısmı hesaba kat
        if y2 < bottom_half_start:
            continue  # Engel tamamen üst yarıda, atla
        
        # Alt yarıya göre kırp
        y1_clipped = max(y1, bottom_half_start)
        y2_clipped = y2
        
        # Sol bölge ile kesişim
        if x1 < left_end:
            x1_region = x1
            x2_region = min(x2, left_end)
            area = (x2_region - x1_region) * (y2_clipped - y1_clipped)
            region_obstacle_area["SOL"] += max(0, area)
        
        # Orta bölge ile kesişim
        if x2 > left_end and x1 < right_start:
            x1_region = max(x1, left_end)
            x2_region = min(x2, right_start)
            area = (x2_region - x1_region) * (y2_clipped - y1_clipped)
            region_obstacle_area["DÜZ"] += max(0, area)
        
        # Sağ bölge ile kesişim
        if x2 > right_start:
            x1_region = max(x1, right_start)
            x2_region = x2
            area = (x2_region - x1_region) * (y2_clipped - y1_clipped)
            region_obstacle_area["SAG"] += max(0, area)
    
    # Doluluk oranlarını hesapla
    density = {}
    for region in ["SOL", "DÜZ", "SAG"]:
        density[region] = region_obstacle_area[region] / region_areas[region] if region_areas[region] > 0 else 0
    
    # Tüm bölgeler threshold üzerinde doluysa DUR
    if all(d >= threshold for d in density.values()):
        return "DUR"
    
    # En az yoğunluklu bölgeyi bul (öncelik: DÜZ > SOL > SAĞ)
    min_density = min(density.values())
    
    # Öncelik sırası: DÜZ (düz gitmek tercih edilir), sonra SOL, sonra SAG
    priority_order = ["DÜZ", "SOL", "SAG"]
    
    for direction in priority_order:
        if density[direction] == min_density and density[direction] < threshold:
            return direction
    
    # Fallback
    return "DUR"


def draw_regions(frame: np.ndarray, direction: str) -> np.ndarray:
    """
    Görüntü üzerine 3 bölgeyi ve yön bilgisini çizer.
    
    Args:
        frame: OpenCV görüntüsü
        direction: Hesaplanan yön
    
    Returns:
        Annotated görüntü
    """
    height, width = frame.shape[:2]
    bottom_half_start = height // 2
    left_end = width // 3
    right_start = 2 * width // 3
    
    # Bölge çizgileri (yarı saydam)
    overlay = frame.copy()
    
    # Sol bölge - Kırmızı tonlu
    cv2.rectangle(overlay, (0, bottom_half_start), (left_end, height), (0, 0, 100), -1)
    # Orta bölge - Yeşil tonlu
    cv2.rectangle(overlay, (left_end, bottom_half_start), (right_start, height), (0, 100, 0), -1)
    # Sağ bölge - Mavi tonlu
    cv2.rectangle(overlay, (right_start, bottom_half_start), (width, height), (100, 0, 0), -1)
    
    # Overlay'i ana görüntüye karıştır
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Bölge sınır çizgileri
    cv2.line(frame, (left_end, bottom_half_start), (left_end, height), (255, 255, 255), 2)
    cv2.line(frame, (right_start, bottom_half_start), (right_start, height), (255, 255, 255), 2)
    cv2.line(frame, (0, bottom_half_start), (width, bottom_half_start), (255, 255, 255), 2)
    
    # Bölge etiketleri
    cv2.putText(frame, "SOL", (left_end // 2 - 30, bottom_half_start + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "ORTA", ((left_end + right_start) // 2 - 35, bottom_half_start + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG", (right_start + (width - right_start) // 2 - 25, bottom_half_start + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Yön bilgisi - Büyük ve belirgin
    direction_colors = {
        "DÜZ": (0, 255, 0),      # Yeşil
        "SOL": (0, 255, 255),    # Sarı
        "SAG": (255, 255, 0),    # Cyan
        "DUR": (0, 0, 255)       # Kırmızı
    }
    color = direction_colors.get(direction, (255, 255, 255))
    
    # Arka plan kutusu
    cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 70), color, 3)
    
    # Yön metni
    cv2.putText(frame, f"Yon: {direction}", (20, 52), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    return frame


def main():
    """
    Ana fonksiyon - Kamerayı açar, YOLO ile engel tespiti yapar ve yön belirler.
    """
    global speech_thread_running
    
    print("YOLOv11 Engel Tespit Sistemi Baslatiliyor...")
    print("Cikmak icin 'q' tusuna basin.")
    print("-" * 50)
    
    # Ses dosyalarını oluştur
    print("Ses dosyalari hazirlaniyor...")
    create_audio_files()
    
    # Ses thread'ini başlat
    print("Ses sistemi baslatiliyor...")
    speech_thread_running = True
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    # Test sesi
    speech_queue.put("HAZIR")
    time.sleep(2)  # Test sesinin bitmesini bekle
    print("Ses sistemi hazir!")
    
    # YOLOv11 modelini yükle (VisionPipeline üzerinden)
    print("Vision Pipeline baslatiliyor...")
    try:
        pipeline = VisionPipeline("../models/yolo11n.pt")
        print("Pipeline hazir!")
    except Exception as e:
        print(f"Pipeline baslatma hatasi: {e}")
        return
    
    # Kamerayı aç
    print("Kamera aciliyor...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        print("Lutfen kameranizin bagli oldugunu kontrol edin.")
        return
    
    # Kamera ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Sistem hazir! Engel tespiti basliyor...")
    print("-" * 50)
    
    frame_count = 0
    last_spoken_direction = None  # Son söylenen yön
    speech_cooldown = 0  # Hemen başla
    danger_cooldown = 0  # Yakın engel uyarısı için cooldown
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alinamadi!")
            break
        
        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Vision Pipeline ile işle (YOLO + Canny + IPM + Free Space)
        combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
        
        # Free Space Maskesini BEV görüntüsüne yeşil overlay olarak ekle
        # Maskenin beyaz olduğu yerleri (boş alanları) yeşil yap
        free_space_overlay = np.zeros_like(bev_view)
        free_space_overlay[free_space_mask > 0] = [0, 255, 0]
        
        # BEV görüntüsü ile karıştır
        bev_combined = cv2.addWeighted(bev_view, 0.7, free_space_overlay, 0.3, 0)
        
        # Tespit edilen engelleri topla
        obstacles = []
        
        for item in pipeline_obstacles:
            x1, y1, x2, y2, class_name, confidence = item
            
            # Tüm tespit edilen nesneler engel kabul edilecek
            obstacles.append((x1, y1, x2, y2))
            
            # Mesafe tahmini (Perspektif tabanlı - y2 kullanılarak)
            distance, dist_category = estimate_distance(y2, frame_height)
            
            # Mesafeye göre renk belirle
            if dist_category == "YAKIN":
                box_color = (0, 0, 255)  # Kırmızı - tehlikeli
            elif dist_category == "ORTA":
                box_color = (0, 165, 255)  # Turuncu - dikkat
            else:
                box_color = (0, 255, 0)  # Yeşil - güvenli
            
            # Bounding box çiz (Combined view üzerine)
            cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
            
            # Etiket (mesafe ile birlikte)
            label = f"{class_name}: {distance:.1f}m"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(combined_view, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(combined_view, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # En yakın engeli bul
        min_distance, closest_category, closest_bbox = get_closest_obstacle(obstacles, frame_height)
        
        # Yön hesapla (Yeni Algoritma: Free Space Maskesi üzerinden)
        direction = pipeline.find_best_direction(free_space_mask)
        # Eski yöntem: direction = get_direction(obstacles, frame_width, frame_height)
        
        # Cooldown azalt
        if speech_cooldown > 0:
            speech_cooldown -= 1
        if danger_cooldown > 0:
            danger_cooldown -= 1
        
        # Yakın engel uyarısı (ayrı cooldown ile)
        if closest_category == "YAKIN" and danger_cooldown <= 0:
            print("UYARI: Çok yakın engel!")
            speech_queue.put("YAKIN")
            danger_cooldown = 60  # 2 saniye cooldown
        
        # Sürekli yön söyleme (her 3 saniyede bir)
        if speech_cooldown <= 0:
            print(f"Ses komutu kuyruğa eklendi: {direction}")
            
            # Kuyruğu temizle ve yeni komutu ekle
            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                except:
                    pass
            speech_queue.put(direction)
            
            last_spoken_direction = direction
            speech_cooldown = 90  # 90 kare (yaklaşık 3 saniye) cooldown
        
        # Bölgeleri ve yönü çiz
        combined_view = draw_regions(combined_view, direction)
        
        # Mesafe bilgisi göster
        if min_distance is not None:
            distance_color = (0, 0, 255) if closest_category == "YAKIN" else (0, 255, 255) if closest_category == "ORTA" else (0, 255, 0)
            cv2.putText(combined_view, f"En Yakin: {min_distance:.1f}m", (10, frame_height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, distance_color, 2)
        else:
            cv2.putText(combined_view, "En Yakin: Yok", (10, frame_height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Engel sayısını göster
        cv2.putText(combined_view, f"Engel Sayisi: {len(obstacles)}", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS göster (her 30 karede bir)
        if frame_count % 30 == 0:
            dist_str = f"{min_distance:.1f}m" if min_distance else "Yok"
            print(f"Kare: {frame_count} | Engel: {len(obstacles)} | Yon: {direction} | Mesafe: {dist_str}")
        
        # Görüntüyü göster
        cv2.imshow("YOLOv11 Engel Tespit - Yon Belirleme", combined_view)
        cv2.imshow("Kus Bakisi (BEV) - Free Space", bev_combined)
        
        # Çıkış için 'q' tuşu
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProgram sonlandiriliyor...")
            break
    
    # Temizlik
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Program basariyla sonlandirildi.")


if __name__ == "__main__":
    main()
