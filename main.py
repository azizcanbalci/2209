import cv2
import numpy as np
# from ultralytics import YOLO  # Artık VisionPipeline içinde
from vision_pipeline import VisionPipeline
from navigation_map import NavigationMap, DepthEstimator
from radar_navigation import RadarNavigation
import threading
from queue import Queue
import time
import os
from gtts import gTTS
import pygame

# --- IP KAMERA İÇİN HIZLANDIRICI SINIF ---
class LatestFrameReader:
    """
    IP Kameralardaki gecikmeyi (lag) önlemek için arka planda sürekli okuma yapar
    ve her zaman en son kareyi verir.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Buffer boyutunu küçültmeyi dene (Backend destekliyorsa)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.ret = False
        
        # İlk kareyi oku
        self.ret, self.latest_frame = self.cap.read()
        if not self.ret:
            print("Hata: Kamera başlatılamadı veya akış yok!")
            self.running = False
            return

        # Okuma thread'ini başlat
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.latest_frame = frame
            # CPU'yu boğmamak için minik bir uyku (opsiyonel, gerekirse kaldırılabilir)
            time.sleep(0.001) 

    def read(self):
        with self.lock:
            return self.ret, self.latest_frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
    
    def isOpened(self):
        return self.cap.isOpened()

    def set(self, prop, value):
        self.cap.set(prop, value)

# Pygame mixer başlat
pygame.mixer.init()

# Global ses kuyruğu
speech_queue = Queue()
speech_thread_running = True

# YÖN STABİLİZASYONU - Kör kullanıcı için kritik
from collections import deque
direction_history = deque(maxlen=10)  # Son 10 yön kararı
stable_direction = "DÜZ"  # Stabil yön (söylenecek)
stability_counter = 0  # Aynı yön kaç kez tekrarlandı
MIN_STABILITY_COUNT = 5  # Yön değişmeden önce minimum tekrar sayısı

# Ses dosyaları için klasör
AUDIO_DIR = "audio_cache"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def create_audio_files():
    """Başlangıçta ses dosyalarını oluştur - Engelli bireyler için optimize edilmiş"""
    commands = {
        # Yön komutları - Net ve anlaşılır
        "SOL": "Sola dönün",
        "DÜZ": "Düz devam edin, yol açık",
        "SAG": "Sağa dönün",
        "HAFIF_SOL": "Hafifçe sola yönelin",
        "HAFIF_SAG": "Hafifçe sağa yönelin",
        
        # Uyarı komutları
        "DUR": "Durun! Önünüzde engel var",
        "YAKIN": "Dikkat! Yakınınızda engel var, yavaşlayın",
        "COK_YAKIN": "Durun! Çok yakın engel",
        
        # Bilgi komutları
        "HAZIR": "Sistem hazır. Yürümeye başlayabilirsiniz",
        "ACIK": "Yol açık, güvenle ilerleyebilirsiniz",
        "ENGEL_YOK": "Önünüzde engel yok"
    }
    
    # Yeni komutları oluştur (mevcut olmayanları)
    for key, text in commands.items():
        filepath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        # Yeni komutlar için veya dosya yoksa oluştur
        needs_create = not os.path.exists(filepath) or key in ["COK_YAKIN", "ACIK", "HAFIF_SOL", "HAFIF_SAG"]
        if needs_create:
            print(f"Ses dosyası oluşturuluyor: {text}")
            try:
                tts = gTTS(text=text, lang='tr')
                tts.save(filepath)
            except Exception as e:
                print(f"Ses oluşturma hatası ({key}): {e}")
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


# ============ KALİBRASYON AYARLARI (KÖR KULLANICI İÇİN) ============
CALIBRATION = {
    # BOYUT EŞİKLERİ - Küçük nesneler UZAK kabul edilir
    "tiny_size": 0.08,      # %8'den küçük = kesinlikle uzak (görmezden gel)
    "small_size": 0.15,     # %15'ten küçük = uzak
    "medium_size": 0.25,    # %25'ten küçük = orta
    "large_size": 0.40,     # %40'tan büyük = yakın
    
    # YAKIN uyarısı için nesne hem BÜYÜK hem de ALTTA olmalı
    "min_size_for_warning": 0.25,  # YAKIN uyarısı için min %25 boyut
    "min_y_ratio_for_close": 0.75, # Ekranın alt %25'inde olmalı
}

def estimate_distance(y2: int, frame_height: int, bbox_height: int = None) -> tuple:
    """
    KÖR KULLANICI İÇİN KALİBRE EDİLMİŞ mesafe tahmini.
    KURAL: Küçük nesne = UZAK. Sadece BÜYÜK + ALTTA = YAKIN.
    """
    y_ratio = y2 / frame_height  # Nesnenin dikey konumu (0=üst, 1=alt)
    
    # Boyut oranı hesapla
    if bbox_height is not None and bbox_height > 0:
        size_ratio = bbox_height / frame_height
    else:
        size_ratio = 0.10  # Bilinmiyorsa küçük say
    
    # ========== BOYUT BAZLI SINIFLANDIRMA ==========
    # KURAL 1: Çok küçük nesne = KESİNLİKLE UZAK (ne olursa olsun)
    if size_ratio < CALIBRATION["tiny_size"]:  # <%8
        return 8.0, "UZAK"
    
    # KURAL 2: Küçük nesne = UZAK
    if size_ratio < CALIBRATION["small_size"]:  # <%15
        return 6.0, "UZAK"
    
    # KURAL 3: Orta-küçük nesne = En fazla ORTA olabilir
    if size_ratio < CALIBRATION["medium_size"]:  # <%25
        if y_ratio > 0.85:  # Çok altta
            return 3.0, "ORTA"
        else:
            return 5.0, "UZAK"
    
    # ========== BÜYÜK NESNELER İÇİN KONUM KONTROLÜ ==========
    # KURAL 4: Büyük nesne + altta = YAKIN
    if size_ratio >= CALIBRATION["large_size"]:  # >%40
        if y_ratio > 0.90:  # Ekranın en altında
            return 0.5, "COK_YAKIN"
        elif y_ratio > 0.80:
            return 1.0, "YAKIN"
        elif y_ratio > 0.65:
            return 2.0, "ORTA"
        else:
            return 3.5, "ORTA"
    
    # KURAL 5: Orta boy nesne (%25-%40)
    if y_ratio > 0.88:  # Çok altta
        return 1.5, "YAKIN"
    elif y_ratio > 0.75:
        return 2.5, "ORTA"
    elif y_ratio > 0.60:
        return 4.0, "ORTA"
    else:
        return 5.0, "UZAK"


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
        # Nesne boyutunu da hesaba kat
        bbox_height = y2 - y1
        distance, category = estimate_distance(y2, frame_height, bbox_height)
        
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


def stabilize_direction(new_direction: str) -> str:
    """
    KÖR KULLANICI İÇİN YÖN STABİLİZASYONU.
    Yön değişikliği için aynı yönün birkaç kez tekrarlanması gerekir.
    Bu, hızlı değişimleri önler ve tutarlı komutlar sağlar.
    
    Args:
        new_direction: Pipeline'dan gelen yeni yön
    
    Returns:
        str: Stabil yön (söylenecek)
    """
    global direction_history, stable_direction, stability_counter
    
    # Yeni yönü history'ye ekle
    direction_history.append(new_direction)
    
    # Son N yönün çoğunluğunu bul (ağırlıklı - son yönler daha önemli)
    if len(direction_history) >= 3:
        # Son 5 yönü say
        recent_directions = list(direction_history)[-5:]
        direction_counts = {}
        for i, d in enumerate(recent_directions):
            # Son yönlere daha fazla ağırlık ver
            weight = 1 + (i * 0.5)  # 1, 1.5, 2, 2.5, 3
            direction_counts[d] = direction_counts.get(d, 0) + weight
        
        # En yaygın yönü bul
        most_common = max(direction_counts, key=direction_counts.get)
        most_common_score = direction_counts[most_common]
        total_score = sum(direction_counts.values())
        
        # Yön değişikliği için %60 çoğunluk gerekli
        if most_common_score / total_score >= 0.60:
            if most_common != stable_direction:
                stability_counter += 1
                # Yön değişikliği için minimum 3 ardışık tutarlılık
                if stability_counter >= MIN_STABILITY_COUNT:
                    stable_direction = most_common
                    stability_counter = 0
                    print(f"[STABİL] Yön değişti: {stable_direction}")
            else:
                stability_counter = 0
    
    return stable_direction


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
    
    # ORTA BÖLGE KONTROLÜ - Sadece ortada engel varsa DUR
    # Eğer orta bölge doluysa VE sol/sağ da doluysa → DUR
    # Eğer orta bölge doluysa AMA sol veya sağ açıksa → Yön değiştir
    
    orta_dolu = density["DÜZ"] >= threshold
    sol_acik = density["SOL"] < threshold
    sag_acik = density["SAG"] < threshold
    
    # Eğer ortada engel var ve hiçbir yön açık değilse → DUR
    if orta_dolu and not sol_acik and not sag_acik:
        return "DUR"
    
    # Eğer ortada engel var ama yan yönler açıksa → Yön değiştir (DUR deme!)
    if orta_dolu:
        if sol_acik and sag_acik:
            # Her iki yön de açık, daha az yoğun olanı seç
            return "SOL" if density["SOL"] <= density["SAG"] else "SAG"
        elif sol_acik:
            return "SOL"
        elif sag_acik:
            return "SAG"
    
    # Orta açıksa → DÜZ git (öncelikli)
    if density["DÜZ"] < threshold:
        return "DÜZ"
    
    # En az yoğunluklu bölgeyi bul (öncelik: SOL > SAĞ)
    if sol_acik:
        return "SOL"
    if sag_acik:
        return "SAG"
    
    # Hiçbiri açık değilse (bu noktaya gelmemeli ama fallback)
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
    # IP Webcam URL
    ip_camera_url = "http://10.31.248.109:8080/video"
    
    # Standart VideoCapture yerine LatestFrameReader kullanıyoruz
    # Bu sınıf arka planda sürekli okuma yaparak gecikmeyi önler
    cap = LatestFrameReader(ip_camera_url)
    
    if not cap.isOpened():
        print(f"HATA: IP Kamera ({ip_camera_url}) acilamadi!")
        print("Varsayilan kamera (0) deneniyor...")
        cap = LatestFrameReader(0)
        if not cap.isOpened():
            print("HATA: Hicbir kamera acilamadi!")
            return
    
    # Kamera ayarları (IP kamerada çalışmayabilir ama yine de kalsın)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 3D Navigasyon Haritası
    nav_map = NavigationMap(grid_size=(80, 80), cell_size=0.1)
    depth_estimator = DepthEstimator()
    
    # RADAR NAVİGASYON SİSTEMİ
    radar = RadarNavigation(radar_size=400)
    
    print("Sistem hazir! Engel tespiti ve RADAR basliyor...")
    print("-" * 50)
    
    frame_count = 0
    last_spoken_direction = None  # Son söylenen yön
    speech_cooldown = 0  # Hemen başla
    danger_cooldown = 0  # Yakın engel uyarısı için cooldown
    direction_change_count = 0  # Yön değişikliği sayacı
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Kare alinamadi! (Yeniden baglaniliyor...)")
            time.sleep(0.1)
            continue
        
        # Görüntüyü 640x480 boyutuna zorla
        frame = cv2.resize(frame, (640, 480))
        
        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Vision Pipeline ile işle
        combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
        
        # Free Space Overlay
        free_space_overlay = np.zeros_like(bev_view)
        free_space_overlay[free_space_mask > 0] = [0, 255, 0]
        bev_combined = cv2.addWeighted(bev_view, 0.7, free_space_overlay, 0.3, 0)
        
        # Engelleri topla
        obstacles = []
        for item in pipeline_obstacles:
            x1, y1, x2, y2, class_name, confidence = item
            
            # Bounding box'ı frame sınırları içinde tut (taşma önleme)
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            # Geçersiz box kontrolü (genişlik/yükseklik 0 veya negatif olmasın)
            if x2 <= x1 or y2 <= y1:
                continue
            
            obstacles.append((x1, y1, x2, y2))
            
            # Mesafe tahmini (bbox boyutu ile birlikte)
            bbox_height = y2 - y1
            distance, dist_category = estimate_distance(y2, frame_height, bbox_height)
            
            # Renk belirleme (COK_YAKIN dahil)
            if dist_category in ["YAKIN", "COK_YAKIN"]:
                box_color = (0, 0, 255)  # Kırmızı
            elif dist_category == "ORTA":
                box_color = (0, 165, 255)  # Turuncu
            else:
                box_color = (0, 255, 0)  # Yeşil
            
            # Bounding box çiz (sınırlar içinde)
            cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
            
            # Label pozisyonu (üstte yer yoksa altta göster)
            label = f"{class_name}: {distance:.1f}m"
            label_y = y1 - 5 if y1 > 20 else y2 + 15
            cv2.putText(combined_view, label, (x1, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # En yakın engeli bul
        min_distance, closest_category, closest_bbox = get_closest_obstacle(obstacles, frame_height)
        
        # Yön hesapla ve STABİLİZE ET (kör kullanıcı için kritik)
        raw_direction = pipeline.find_best_direction(free_space_mask)
        direction = stabilize_direction(raw_direction)  # Stabil yön
        
        # Cooldown azalt
        if speech_cooldown > 0:
            speech_cooldown -= 1
        if danger_cooldown > 0:
            danger_cooldown -= 1
        
        # --- KÖR KULLANICI İÇİN OPTİMİZE SESLİ UYARI SİSTEMİ ---
        # UZUN COOLDOWN'LAR - Sakin ve anlaşılır komutlar
        
        # Ortada engel var mı kontrol et (DUR için gerekli)
        orta_engel_var = False
        if closest_bbox is not None:
            cx = (closest_bbox[0] + closest_bbox[2]) // 2  # Engel merkezi X
            left_end = frame_width // 3
            right_start = 2 * frame_width // 3
            orta_engel_var = left_end <= cx <= right_start
        
        # 1. ACİL DURUM: ÇOK YAKIN + ORTADA (en yüksek öncelik)
        if closest_category == "COK_YAKIN" and orta_engel_var and danger_cooldown <= 0:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put("COK_YAKIN")
            danger_cooldown = 90   # 3 saniye - acil uyarılar arası
            speech_cooldown = 90
            print(f"ACİL: Çok yakın engel ORTADA! ({min_distance:.1f}m)")
        
        # 2. DUR komutu (sadece ORTADA engel varsa ve tüm yönler kapalıysa)
        elif raw_direction == "DUR" and orta_engel_var and danger_cooldown <= 0:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put("DUR")
            danger_cooldown = 90   # 3 saniye
            speech_cooldown = 90
            print("ACİL: DUR komutu - Ortada engel!")
        
        # 3. YAKIN ENGEL UYARISI (sadece ORTADA yakın engel varsa)
        elif closest_category == "YAKIN" and orta_engel_var and min_distance is not None and min_distance < 2.0 and danger_cooldown <= 0:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put("YAKIN")
            danger_cooldown = 120  # 4 saniye
            speech_cooldown = 120
            print(f"UYARI: Yakın engel ORTADA ({min_distance:.1f}m)")
        
        # 4. YÖN DEĞİŞİKLİĞİ (stabilize edilmiş - yavaş değişim)
        elif direction != last_spoken_direction and speech_cooldown <= 0:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put(direction)
            last_spoken_direction = direction
            speech_cooldown = 150  # 5 saniye - yön değişiklikleri arası
            print(f"Yön: {direction}")
        
        # 5. YOL AÇIK BİLDİRİMİ (engel yokken)
        elif len(obstacles) == 0 and speech_cooldown <= 0 and last_spoken_direction != "ACIK":
            speech_queue.put("ACIK")
            last_spoken_direction = "ACIK"
            speech_cooldown = 180  # 6 saniye
            print("Bilgi: Yol açık")
        
        # 6. PERİYODİK HATIRLATMA (her 5 saniyede)
        elif speech_cooldown <= 0 and raw_direction not in ["DUR"] and direction not in ["ACIK"]:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put(direction)
            speech_cooldown = 150  # 5 saniye
        
        # === RADAR NAVİGASYON SİSTEMİ ===
        # YOLO engellerini radar'a aktar (mesafe bilgisi ile)
        radar_obstacles = []
        for item in pipeline_obstacles:
            x1, y1, x2, y2, class_name, confidence = item
            bbox_height = y2 - y1
            dist, _ = estimate_distance(y2, frame_height, bbox_height)
            radar_obstacles.append((x1, y1, x2, y2, class_name, dist))
        
        # Radar'ı güncelle ve görselleştir
        radar_img, radar_direction, radar_info = radar.process_frame(
            radar_obstacles, 
            frame_width, 
            frame_height
        )
        
        # Eski navigasyon haritası (isteğe bağlı)
        nav_obstacles = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in obstacles]
        nav_vis, nav_command, nav_obstacles_info = nav_map.process_frame(
            nav_obstacles, 
            free_space_mask, 
            frame_width, 
            frame_height
        )
        
        # Görselleştirme
        combined_view = draw_regions(combined_view, direction)
        
        # Bilgi göster
        if min_distance is not None:
            dist_color = (0, 0, 255) if closest_category in ["YAKIN", "COK_YAKIN"] else (0, 255, 255) if closest_category == "ORTA" else (0, 255, 0)
            cv2.putText(combined_view, f"Mesafe: {min_distance:.1f}m", (10, frame_height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
        
        cv2.putText(combined_view, f"Engel: {len(obstacles)}", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Radar yön bilgisi ana ekrana ekle
        cv2.putText(combined_view, f"Radar: {radar_direction}", (frame_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Debug (her 30 karede)
        if frame_count % 30 == 0:
            dist_str = f"{min_distance:.1f}m" if min_distance else "Yok"
            print(f"Kare: {frame_count} | Engel: {len(obstacles)} | Radar: {radar_direction} | Stabil: {direction}")
        
        # Görüntüleri göster
        cv2.imshow("Akilli Asistan - Yonlendirme", combined_view)
        cv2.imshow("Kus Bakisi (BEV)", bev_combined)
        cv2.imshow("RADAR Navigasyon", radar_img)
        
        # Çıkış
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
