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
from services.voice_chat import VoiceChatAgent

# --- AYARLAR ---
GEMINI_API_KEY = "BURAYA_API_KEY_GELECEK" # Kullanıcıdan alınacak

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
    KALİBRASYON: Ortalama bir insan boyu ve kamera açısı varsayılarak optimize edildi.
    """
    # Nesnenin alt noktasının frame'e oranı (0.0 - 1.0)
    # 1.0 = Ekranın en altı (Ayak ucu) -> 0 metre
    # 0.5 = Ekranın ortası -> Sonsuz (Ufuk)
    ratio = y2 / frame_height
    
    # Mesafe Tahmini (Empirik Formül)
    # Basit Eşikler (Optimize Edilmiş)
    if ratio > 0.90:      # < 0.5 metre (Çok Tehlikeli)
        distance = 0.3
        category = "YAKIN"
    elif ratio > 0.75:    # 0.5 - 1.5 metre (Tehlikeli)
        distance = 1.0
        category = "YAKIN"
    elif ratio > 0.60:    # 1.5 - 3.0 metre (Dikkat)
        distance = 2.5
        category = "ORTA"
    elif ratio > 0.45:    # 3.0 - 6.0 metre (Güvenli)
        distance = 5.0
        category = "ORTA"
    else:                 # > 6.0 metre (Uzak)
        distance = 10.0
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
    Ana fonksiyon - İki modlu sistem:
    1. Yol Modu (VisionPipeline)
    2. Sohbet Modu (VoiceChatAgent)
    """
    global speech_thread_running
    
    print("Akıllı Asistan Sistemi Başlatılıyor...")
    print("Çıkmak için 'q' tuşuna basın.")
    print("-" * 50)
    
    # Ses dosyalarını oluştur
    create_audio_files()
    
    # Ses thread'ini başlat (Yol modu için)
    speech_thread_running = True
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    # Sesli Sohbet Ajanı (Başlangıçta pasif)
    voice_agent = None
    if GEMINI_API_KEY != "BURAYA_API_KEY_GELECEK":
        voice_agent = VoiceChatAgent(GEMINI_API_KEY)
    else:
        print("UYARI: Gemini API Key girilmedi! Sohbet modu çalışmayacak.")

    # YOLOv11 modelini yükle (VisionPipeline üzerinden)
    print("Vision Pipeline başlatılıyor...")
    try:
        pipeline = VisionPipeline("../models/yolo11n.pt")
        print("Pipeline hazır!")
    except Exception as e:
        print(f"Pipeline başlatma hatası: {e}")
        return
    
    # Kamerayı aç
    print("Kamera açılıyor...")
    ip_camera_url = "http://172.18.161.201:8080/video"
    cap = LatestFrameReader(ip_camera_url)
    
    if not cap.isOpened():
        print(f"HATA: IP Kamera ({ip_camera_url}) açılamadı!")
        print("Varsayılan kamera (0) deneniyor...")
        cap = LatestFrameReader(0)
        if not cap.isOpened():
            print("HATA: Hiçbir kamera açılamadı!")
            return
    
    print("Sistem hazır!")
    
    # --- MOD SEÇİMİ (BAŞLANGIÇ) ---
    print("\n" + "="*30)
    print("LÜTFEN BİR MOD SEÇİNİZ:")
    print("[1] Yol Modu (Kamera + Engel Tespiti)")
    print("[2] Sohbet Modu (Yapay Zeka Asistanı)")
    print("="*30)
    
    while True:
        user_choice = input("Seçiminiz (1 veya 2): ").strip()
        if user_choice == '1':
            current_mode = 1
            print(">> YOL MODU SEÇİLDİ.")
            break
        elif user_choice == '2':
            current_mode = 2
            print(">> SOHBET MODU SEÇİLDİ.")
            break
        else:
            print("Hatalı seçim! Lütfen 1 veya 2 yazıp Enter'a basın.")

    speech_queue.put("HAZIR") # "Sistem hazır" sesi
    if current_mode == 2 and voice_agent:
        voice_agent.start_session()
    
    frame_count = 0
    speech_cooldown = 0
    danger_cooldown = 0
    
    while True:
        # Klavye kontrolü (Çalışma anında değişim için)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if current_mode != 1:
                current_mode = 1
                print("MOD DEĞİŞTİ: Yol Modu")
                if voice_agent: voice_agent.stop_session()
                speech_queue.put("Yol modu aktif")
        elif key == ord('2'):
            if current_mode != 2:
                current_mode = 2
                print("MOD DEĞİŞTİ: Sohbet Modu")
                if voice_agent: voice_agent.start_session()
                else: speech_queue.put("Sohbet modu kullanılamıyor")
        
        # --- MOD 1: YOL MODU ---
        if current_mode == 1:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            
            # Vision Pipeline İşlemleri
            combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
            
            # Free Space Overlay
            free_space_overlay = np.zeros_like(bev_view)
            free_space_overlay[free_space_mask > 0] = [0, 255, 0]
            bev_combined = cv2.addWeighted(bev_view, 0.7, free_space_overlay, 0.3, 0)
            
            # Engelleri Topla ve Çiz
            obstacles = []
            for item in pipeline_obstacles:
                x1, y1, x2, y2, class_name, confidence = item
                obstacles.append((x1, y1, x2, y2))
                
                # Mesafe ve Çizim
                distance, dist_category = estimate_distance(y2, frame_height)
                box_color = (0, 0, 255) if dist_category == "YAKIN" else (0, 165, 255) if dist_category == "ORTA" else (0, 255, 0)
                cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(combined_view, f"{class_name} {distance:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # En yakın engel ve Yön
            min_distance, closest_category, closest_bbox = get_closest_obstacle(obstacles, frame_height)
            direction = pipeline.find_best_direction(free_space_mask)
            
            # Sesli Uyarılar (Yol Modu)
            if speech_cooldown > 0: speech_cooldown -= 1
            if danger_cooldown > 0: danger_cooldown -= 1
            
            if closest_category == "YAKIN" and danger_cooldown <= 0:
                speech_queue.put("YAKIN")
                danger_cooldown = 60
            
            if speech_cooldown <= 0:
                # Kuyruğu temizle (gecikmeyi önle)
                while not speech_queue.empty():
                    try: speech_queue.get_nowait()
                    except: pass
                speech_queue.put(direction)
                speech_cooldown = 90
            
            # Görselleştirme
            combined_view = draw_regions(combined_view, direction)
            cv2.imshow("Akilli Asistan", combined_view)
            # cv2.imshow("Kus Bakisi", bev_combined)

        # --- MOD 2: SOHBET MODU ---
        elif current_mode == 2:
            # Sohbet modunda kamerayı göstermeye devam et ama işlem yapma
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (640, 480))
                # Bilgi ekranı
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                cv2.putText(frame, "SOHBET MODU AKTIF", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Konusmak icin bekleyin...", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, "[1] Yol Moduna Don", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow("Akilli Asistan", frame)
            
            # Sesli Sohbet İşlemi (Tek adım)
            if voice_agent:
                voice_agent.process_step()
    
    # Temizlik
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")


if __name__ == "__main__":
    main()
