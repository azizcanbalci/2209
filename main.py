# === ALSA/JACK LOG KİRLİLİĞİNİ KAPAT ===
# PyAudio/PortAudio kullanırken oluşan gereksiz ALSA/JACK uyarılarını susturur
import os
import sys
import ctypes

# ALSA hata mesajlarını sustur (Linux)
try:
    # libasound2 yükle ve hata handler'ını değiştir
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass  # Hiçbir şey yapma - sessiz kal
    
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass  # Linux değilse veya libasound yoksa atla

# JACK hata mesajlarını sustur - stderr'i /dev/null'a yönlendir
# Bu daha agresif ama JACK mesajlarını susturmanın tek yolu
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

# Orijinal stderr'i sakla (gerekirse geri almak için)
_original_stderr = sys.stderr

def suppress_audio_errors():
    """Ses kütüphanelerinin stderr çıktılarını sustur"""
    try:
        # /dev/null'a yönlendir (Linux)
        devnull = open('/dev/null', 'w')
        os.dup2(devnull.fileno(), 2)  # stderr = fd 2
    except:
        pass

def restore_stderr():
    """Orijinal stderr'i geri yükle"""
    sys.stderr = _original_stderr

# Pygame ve PyAudio yüklemeden önce hataları sustur
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import cv2
import numpy as np
# from ultralytics import YOLO  # Artık VisionPipeline içinde
from vision_pipeline import VisionPipeline
from navigation_map import NavigationMap, DepthEstimator
from radar_navigation import RadarNavigation
import threading
from queue import Queue
import time
from gtts import gTTS

# Ses kütüphaneleri import edilmeden önce JACK hatalarını sustur
suppress_audio_errors()

import pygame
import uuid
import tempfile

# Servisler
from services.ocr_reader import ocr_reader, read_text
from services.object_describer import describe_objects, get_turkish_name
from services.object_searcher import search_object, get_available_objects
from services import voice_chat  # MOD 5: Sesli AI Sohbet
from services import image_qa  # MOD 6: Görsel Soru-Cevap (Gemini)
from services import slam_mapper  # MOD 7: 3D Harita (SLAM)
from services import voice_command  # Sesli Komut Sistemi

# --- RASPBERRY PI CAMERA MODULE V3 İÇİN SINIF ---
# 64-bit Bookworm için Picamera2 kullanır
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("UYARI: Picamera2 bulunamadi. USB/IP kamera kullanilacak.")


class PiCameraReader:
    """
    Raspberry Pi Camera Module v3 için optimized reader.
    Picamera2 kullanarak OpenCV uyumlu frame'ler sağlar.
    """
    def __init__(self, camera_num=0, width=1280, height=720):
        self.width = width
        self.height = height
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("Picamera2 kurulu değil!")
        
        try:
            # Picamera2 başlat (camera_num: 0 veya 1)
            self.picam2 = Picamera2(camera_num)
            
            # Kamera yapılandırması
            config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"},
                buffer_count=2
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            # === AUTOFOCUS AYARLARI (Pi Camera Module v3) ===
            try:
                from libcamera import controls
                # Surekli autofocus - kamera surekli odaklanir
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfSpeed": controls.AfSpeedEnum.Fast
                })
                print("[OK] Autofocus: Surekli mod aktif")
            except Exception as e:
                print(f"[UYARI] Autofocus ayarlanamadi: {e}")
            
            # İlk kareyi al - autofocus icin biraz daha bekle
            time.sleep(1.0)  # Kamera + autofocus stabilizasyonu
            self.latest_frame = self.picam2.capture_array()
            self.running = True
            
            # Arka plan thread'i başlat
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print(f"[OK] Pi Camera {camera_num} baslatildi ({width}x{height})")
            
        except Exception as e:
            print(f"[HATA] Pi Camera baslatma hatasi: {e}")
            self.running = False
            raise
    
    def _update(self):
        """Arka planda sürekli kare yakala"""
        while self.running:
            try:
                frame = self.picam2.capture_array()
                # RGB olarak kalsin (donusum yok)
                with self.lock:
                    self.latest_frame = frame
            except Exception as e:
                print(f"Kare yakalama hatası: {e}")
                time.sleep(0.01)
    
    def read(self):
        """En son kareyi döndür (OpenCV uyumlu)"""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None
    
    def release(self):
        """Kamerayı kapat"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, 'picam2'):
            self.picam2.stop()
            self.picam2.close()
        print("Pi Camera kapatıldı.")
    
    def isOpened(self):
        return self.running
    
    def set(self, prop, value):
        # OpenCV property setleri için placeholder
        pass


class LatestFrameReader:
    """
    Fallback: USB/IP Kamera için threaded reader.
    Pi Camera yoksa veya hata olursa kullanılır.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.ret = False
        
        self.ret, self.latest_frame = self.cap.read()
        if not self.ret:
            print("Hata: Kamera başlatılamadı!")
            self.running = False
            return

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.latest_frame = frame
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

# === AKILLI HAFIZA SİSTEMİ ===
class SmartNavigator:
    """
    Akıllı navigasyon hafızası - gerçek zamanlı yönlendirme için
    """
    def __init__(self):
        self.direction_history = deque(maxlen=15)  # Son 15 yön
        self.last_command = None
        self.last_command_time = 0
        self.command_count = {}  # Komut sayaçları
        self.movement_state = "IDLE"  # IDLE, MOVING, TURNING
        self.turn_direction = None  # Hangi yöne dönülüyor
        self.consecutive_same = 0  # Aynı komut kaç kez üst üste geldi
        
        # Zaman bazlı ayarlar (saniye cinsinden frame sayısı, 30fps varsayım)
        self.min_command_interval = 45  # 1.5 saniye - komutlar arası minimum süre
        self.urgent_interval = 15  # 0.5 saniye - acil durumlar için
        self.direction_change_threshold = 8  # Yön değişimi için gereken tutarlılık
        
    def add_direction(self, direction):
        """Yeni yön ekle ve analiz et"""
        self.direction_history.append(direction)
        
        # Aynı yön kaç kez tekrarlandı?
        if len(self.direction_history) >= 2:
            if self.direction_history[-1] == self.direction_history[-2]:
                self.consecutive_same += 1
            else:
                self.consecutive_same = 1
        
    def get_dominant_direction(self):
        """Son yönlerin baskın olanını bul"""
        if len(self.direction_history) < 3:
            return self.direction_history[-1] if self.direction_history else "DÜZ"
        
        from collections import Counter
        recent = list(self.direction_history)[-10:]  # Son 10
        counts = Counter(recent)
        
        # En çok tekrar eden yön
        most_common = counts.most_common(1)[0]
        
        # %40 çoğunluk gerekli
        if most_common[1] >= len(recent) * 0.4:
            return most_common[0]
        
        return self.direction_history[-1]
    
    def should_speak(self, direction, frame_count, is_urgent=False):
        """
        Bu komutu söylemeli miyiz?
        Akıllı karar mekanizması
        """
        current_time = frame_count
        time_since_last = current_time - self.last_command_time
        
        # Acil durum (DUR, COK_YAKIN)
        if is_urgent:
            if time_since_last >= self.urgent_interval:
                return True
            return False
        
        # Normal komut
        if time_since_last < self.min_command_interval:
            return False
        
        # Yön değişikliği kontrolü
        if direction != self.last_command:
            # Yeterince tutarlı mı?
            if self.consecutive_same >= self.direction_change_threshold:
                return True
            # Çok farklı bir yön mü? (örn: SOL'dan SAG'a)
            opposite_pairs = [("SOL", "SAG"), ("HAFIF_SOL", "HAFIF_SAG")]
            for pair in opposite_pairs:
                if (self.last_command in pair and direction in pair and 
                    self.last_command != direction):
                    # Zıt yönler - daha fazla tutarlılık iste
                    if self.consecutive_same >= self.direction_change_threshold + 3:
                        return True
                    return False
            return True
        
        # Aynı komut - periyodik hatırlatma
        if time_since_last >= self.min_command_interval * 2:  # 3 saniye
            return True
        
        return False
    
    def update_state(self, direction, frame_count):
        """Durumu güncelle ve söylenecek komutu döndür"""
        self.add_direction(direction)
        dominant = self.get_dominant_direction()
        
        is_urgent = dominant in ["DUR", "COK_YAKIN"]
        
        if self.should_speak(dominant, frame_count, is_urgent):
            self.last_command = dominant
            self.last_command_time = frame_count
            return dominant
        
        return None

# Global navigator
smart_nav = SmartNavigator()

# === MOD YÖNETİCİSİ ===
class ModeManager:
    """
    7 Modlu Asistan Sistemi:
    1 - Navigasyon Modu (varsayılan)
    2 - Metin Okuma Modu (PaddleOCR)
    3 - Nesne Tanıma Modu (YOLO detaylı)
    4 - Nesne Arama Modu
    5 - Sesli AI Sohbet Modu (Mistral)
    6 - Görsel Soru-Cevap Modu (Gemini)
    7 - 3D Haritalama Modu (SLAM)
    """
    MODES = {
        1: "NAVİGASYON",
        2: "METİN OKUMA",
        3: "NESNE TANIMA",
        4: "NESNE ARAMA",
        5: "SESLİ AI SOHBET",
        6: "GÖRSEL SORU-CEVAP",
        7: "3D HARİTALAMA"
    }
    
    def __init__(self):
        self.current_mode = 1
        self.search_target = None  # Mod 4 için aranan nesne
        self.last_ocr_time = 0
        self.last_describe_time = 0
        self.ocr_cooldown = 90  # 3 saniye
        self.describe_cooldown = 60  # 2 saniye
        
    def switch_mode(self, mode_num):
        """Mod değiştir"""
        if mode_num in self.MODES:
            self.current_mode = mode_num
            return self.MODES[mode_num]
        return None
    
    def get_mode_name(self):
        """Mevcut mod adını döndür"""
        return self.MODES.get(self.current_mode, "UNKNOWN")

# Global mod yöneticisi
mode_manager = ModeManager()

# === GEÇİCİ SES DOSYASI FONKSİYONLARI ===
# Ses için lock (thread-safe)
_speech_lock = threading.Lock()

def speak_text_temp(text, lang='tr'):
    """
    Metni geçici ses dosyasına çevirip seslendir, sonra sil
    """
    if not text or len(text.strip()) == 0:
        return
    
    temp_file = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4().hex[:8]}.mp3")
    try:
        # gTTS ile ses dosyası oluştur
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)
        
        # Thread-safe ses çalma
        with _speech_lock:
            # Mixer başlatılmamışsa başlat
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
        
        time.sleep(0.1)
        
        # Dosyayı sil
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        print(f"[SES] Seslendirme tamamlandi: {text[:50]}...")
    except Exception as e:
        print(f"[HATA] Ses hatasi: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def speak_text_async(text, lang='tr'):
    """Metni arka planda seslendir"""
    thread = threading.Thread(target=speak_text_temp, args=(text, lang), daemon=True)
    thread.start()

# Ses dosyaları için klasör
AUDIO_DIR = "audio_cache"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def create_audio_files():
    """Başlangıçta ses dosyalarını oluştur - Engelli bireyler için optimize edilmiş"""
    commands = {
        # Yön komutları - Kısa ve net
        "SOL": "Sola",
        "DÜZ": "Düz",
        "SAG": "Sağa",
        "HAFIF_SOL": "Hafif sola",
        "HAFIF_SAG": "Hafif sağa",
        
        # Hareket komutları
        "ILERLE": "İlerle",
        "DEVAM": "Devam",
        
        # Uyarı komutları - Acil ve net
        "DUR": "Dur!",
        "YAKIN": "Dikkat! Engel yakın",
        "COK_YAKIN": "Dur! Engel çok yakın",
        
        # === KAÇIŞ YÖNLÜ ACİL KOMUTLAR ===
        "DUR_SOL": "Dur! Sola kaç",
        "DUR_SAG": "Dur! Sağa kaç",
        "DUR_GERI": "Dur! Geri çekil",
        "YAKIN_SOL": "Dikkat! Sola yönel",
        "YAKIN_SAG": "Dikkat! Sağa yönel",
        "ENGEL_SOL": "Engel solda, sağa git",
        "ENGEL_SAG": "Engel sağda, sola git",
        "ENGEL_ORTA": "Engel önde",
        
        # === MOD BİLDİRİMLERİ ===
        "MOD_1": "Navigasyon modu aktif",
        "MOD_2": "Metin okuma modu aktif",
        "MOD_3": "Nesne tanıma modu aktif",
        "MOD_4": "Arama modu aktif. Aramak istediğiniz nesneyi söyleyin",
        "MOD_5": "Sesli sohbet modu aktif",
        "MOD_6": "Görsel soru cevap modu aktif",
        "MOD_7": "Üç boyutlu haritalama modu aktif",
        "METIN_YOK": "Metin bulunamadı",
        "NESNE_YOK": "Görüş alanında nesne yok",
        "BULUNAMADI": "Aranan nesne bulunamadı",
        
        # Bilgi komutları
        "HAZIR": "Sistem hazır",
        "ACIK": "Yol açık",
        "ENGEL_YOK": "Engel yok"
    }
    
    # Tüm ses dosyalarını yeniden oluştur (kısa ve öz komutlar)
    for key, text in commands.items():
        filepath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        print(f"Ses dosyası oluşturuluyor: {key} -> {text}")
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
            # Mixer başlatılmamışsa başlat
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Ses çalma hatası: {e}")
            # Mixer'ı yeniden başlatmayı dene
            try:
                pygame.mixer.quit()
                pygame.mixer.init()
            except:
                pass


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
                    # Debug mesajı sadece MOD 1'de göster
                    # Debug mesajı sadece Navigasyon modunda (MOD 1)
                    if mode_manager.current_mode == 1:
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


def run_voice_chat_mode():
    """
    MOD 5: Sesli AI Sohbet Modu
    Kamera kullanmadan, sadece sesli konuşma ile AI sohbeti
    """
    print("\n" + "=" * 60)
    print("    SESLI AI SOHBET MODU")
    print("=" * 60)
    
    # Voice chat servisini başlat
    if not voice_chat.init():
        print("[HATA] Sesli sohbet baslatilamadi!")
        speak_text_async("Sesli sohbet başlatılamadı. Token veya internet bağlantısını kontrol edin.")
        return
    
    print("\nKOMUTLAR:")
    print("  - Konuşarak soru sorun")
    print("  - 'kapat' veya 'çıkış' diyerek çıkın")
    print("  - Ctrl+C ile acil çıkış")
    print("-" * 60)
    
    # Hoşgeldin mesajı
    speak_text_async("Merhaba! Size nasıl yardımcı olabilirim?")
    time.sleep(2)
    
    try:
        while True:
            print("\n[MIC] Dinliyorum... (Konusabilirsiniz)")
            
            # Dinle
            success, text = voice_chat.listen(timeout=7, phrase_limit=20)
            
            if not success:
                if text is None:
                    # Timeout - sessizlik
                    continue
                elif text == "":
                    # Anlaşılamadı
                    speak_text_async("Sizi anlayamadım, tekrar eder misiniz?")
                    continue
                else:
                    # Hata mesajı
                    print(f"[HATA] {text}")
                    continue
            
            print(f"[SIZ] {text}")
            
            # Çıkış komutu kontrolü
            if voice_chat.is_exit_command(text):
                speak_text_async("Görüşmek üzere, hoşça kalın!")
                time.sleep(2)
                break
            
            # AI'a sor
            print("[BEKLENIYOR] Dusunuyorum...")
            answer = voice_chat.ask(text)
            
            # Cevabı seslendir
            print(f"[AI] {answer}")
            speak_text_async(answer)
            
            # Cevap bitmesini bekle
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n[UYARI] Kullanici tarafindan durduruldu.")
        speak_text_async("Görüşürüz!")
        time.sleep(1)
    
    print("\n[OK] Sesli sohbet sonlandirildi.")


def main():
    """
    Ana fonksiyon - 7 Modlu Görme Engelli Asistanı
    Mod 1: Navigasyon (varsayılan)
    Mod 2: Metin Okuma (PaddleOCR)
    Mod 3: Nesne Tanıma
    Mod 4: Nesne Arama
    Mod 5: Sesli AI Sohbet (Mistral)
    Mod 6: Görsel Soru-Cevap (Gemini)
    Mod 7: 3D Haritalama (SLAM)
    """
    global speech_thread_running, mode_manager
    
    print("=" * 60)
    print("    GÖRME ENGELLİ ASİSTANI - 7 MODLU SİSTEM")
    print("=" * 60)
    print("MOD KONTROLLERI:")
    print("  1 - Navigasyon Modu (yön komutları)")
    print("  2 - Metin Okuma Modu (OCR)")
    print("  3 - Nesne Tanıma Modu (çevredeki nesneler)")
    print("  4 - Nesne Arama Modu")
    print("  5 - Sesli AI Sohbet Modu (Mistral)")
    print("  6 - Görsel Soru-Cevap Modu (Gemini)")
    print("  7 - 3D Haritalama Modu (SLAM)")
    print("  q - Çıkış")
    print("-" * 60)
    
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
    time.sleep(2)
    print("Ses sistemi hazir!")
    
    # === BAŞLANGIÇ MOD SEÇİMİ ===
    print("\n" + "=" * 60)
    print("    BAŞLANGIÇ MODU SEÇİN")
    print("=" * 60)
    print("  1 - Navigasyon Modu (yön komutları)")
    print("  2 - Metin Okuma Modu (OCR)")
    print("  3 - Nesne Tanıma Modu (çevredeki nesneler)")
    print("  4 - Nesne Arama Modu")
    print("  5 - Sesli AI Sohbet Modu (Mistral)")
    print("  6 - Görsel Soru-Cevap Modu (Gemini)")
    print("  7 - 3D Haritalama Modu (SLAM)")
    print("-" * 60)
    
    while True:
        try:
            mod_secimi = input("Mod numarası girin (1-7): ").strip()
            if mod_secimi in ['1', '2', '3', '4', '5', '6', '7']:
                mode_manager.switch_mode(int(mod_secimi))
                break
            else:
                print("Geçersiz seçim! 1-7 arası girin.")
        except:
            print("Geçersiz giriş!")
    
    print(f"\n[OK] {mode_manager.get_mode_name()} MODU SECILDI!")
    speech_queue.put(f"MOD_{mode_manager.current_mode}")
    time.sleep(1)
    
    # Mod 4 için başlangıçta arama hedefi al
    search_target = None
    if mode_manager.current_mode == 4:
        print("\nAramak istediğiniz nesneyi yazın:")
        search_target = input("Aranacak nesne: ").strip()
        if search_target:
            print(f"[ARAMA] '{search_target}' aranacak...")
            speak_text_async(f"{search_target} aranıyor")
    
    # MOD 5: Sesli AI Sohbet - Ayrı döngüde çalışır (kamera gerektirmez)
    if mode_manager.current_mode == 5:
        run_voice_chat_mode()
        return  # Sohbet bitince program sonlanır
    
    # OCR sadece MOD 2'de lazy loading ile yüklenecek
    # Başlangıçta yükleme yapılmıyor
    
    # MOD 7: SLAM başlangıç mesajı
    if mode_manager.current_mode == 7:
        print("\n3D HARITALAMA MODU")
        print("Kontroller:")
        print("  SPACE - Haritayı kaydet (maps/ klasörüne)")
        print("  L     - Harita yükle")
        print("  R     - Haritayı sıfırla")
        print("  I     - İstatistikleri göster")
        slam_mapper.init()
        print("SLAM sistemi hazır!")
    
    print("-" * 60)
    
    # YOLOv11 modelini yükle (VisionPipeline üzerinden)
    print("Vision Pipeline baslatiliyor...")
    try:
        pipeline = VisionPipeline("../models/yolo11n.pt")
        print("Pipeline hazir!")
    except Exception as e:
        print(f"Pipeline baslatma hatasi: {e}")
        return
    
    # Kamerayı aç - Raspberry Pi Camera Module v3 (Port 0)
    print("Kamera açılıyor...")
    
    cap = None
    
    # Önce Pi Camera dene (Raspberry Pi'de)
    if PICAMERA_AVAILABLE:
        try:
            print("Pi Camera Module v3 deneniyor (Port 0)...")
            cap = PiCameraReader(camera_num=0, width=640, height=480)
            print("[OK] Pi Camera basariyla baslatildi!")
        except Exception as e:
            print(f"[UYARI] Pi Camera baslatilamadi: {e}")
            cap = None
    
    # Pi Camera yoksa USB kamera dene
    if cap is None or not cap.isOpened():
        print("USB/Varsayılan kamera (0) deneniyor...")
        cap = LatestFrameReader(0)
        if not cap.isOpened():
            print("HATA: Hiçbir kamera açılamadı!")
            return
        print("[OK] USB kamera baslatildi.")
    
    # 3D Navigasyon Haritası
    nav_map = NavigationMap(grid_size=(80, 80), cell_size=0.1)
    depth_estimator = DepthEstimator()
    
    # RADAR NAVİGASYON SİSTEMİ
    radar = RadarNavigation(radar_size=400)
    
    print("Sistem hazir! 4 MODLU ASİSTAN başlıyor...")
    print(f"Aktif Mod: {mode_manager.get_mode_name()}")
    print("-" * 50)
    
    frame_count = 0
    last_spoken_direction = None
    speech_cooldown = 0
    danger_cooldown = 0
    direction_change_count = 0
    
    # search_target başlangıçta tanımlandı (mod 4 seçiliyse)
    search_found = False
    
    # Mod zamanlayıcıları
    last_ocr_time = 0
    last_describe_time = 0
    last_search_time = 0
    
    # MOD'a göre pencere başlıkları
    def close_all_windows():
        cv2.destroyAllWindows()
    
    while True:
        current_mode = mode_manager.current_mode
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Kare alinamadi! (Yeniden baglaniliyor...)")
            time.sleep(0.1)
            continue
        
        # Görüntüyü 640x480 boyutuna zorla
        frame = cv2.resize(frame, (640, 480))
        
        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # ============================================================
        # MOD 7: 3D HARİTALAMA (SLAM) - Frame alındıktan hemen sonra
        # ============================================================
        if current_mode == 7:
            # SLAM frame işle
            success = slam_mapper.process_frame(frame)
            
            # SLAM görselleştirmesi al (frame üzerine çizim)
            slam_vis = slam_mapper.get_visualization(frame)
            
            # Kuş bakışı harita al
            topdown = slam_mapper.get_topdown_map()
            
            # İstatistikleri al
            stats = slam_mapper.get_stats()
            
            # Pencereleri göster
            cv2.imshow("SLAM Kamera", slam_vis)
            cv2.imshow("SLAM Harita", topdown)
            
            # Tuş kontrolü MOD 7 için
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Haritayı kaydet
                maps_dir = os.path.join(os.path.dirname(__file__), "maps")
                os.makedirs(maps_dir, exist_ok=True)
                filepath = os.path.join(maps_dir, f"room_map_{int(time.time())}.ply")
                if slam_mapper.save_map(filepath):
                    print(f"[OK] Harita kaydedildi: {filepath}")
                    speak_text_async("Harita kaydedildi")
                else:
                    print("[HATA] Harita kaydedilemedi (yeterli nokta yok)")
                    speak_text_async("Harita kaydedilemedi")
            elif key == ord('l'):  # Harita yükle
                maps_dir = os.path.join(os.path.dirname(__file__), "maps")
                if os.path.exists(maps_dir):
                    ply_files = [f for f in os.listdir(maps_dir) if f.endswith('.ply')]
                    if ply_files:
                        latest = sorted(ply_files)[-1]
                        filepath = os.path.join(maps_dir, latest)
                        if slam_mapper.load_map(filepath):
                            print(f"[OK] Harita yuklendi: {latest}")
                            speak_text_async("Harita yüklendi")
                        else:
                            print("[HATA] Harita yuklenemedi")
                    else:
                        print("[HATA] Kayitli harita bulunamadi")
                        speak_text_async("Kayıtlı harita yok")
            elif key == ord('r'):  # Haritayı sıfırla
                slam_mapper.reset()
                print("SLAM sifirlanadi")
                speak_text_async("Harita sıfırlandı")
            elif key == ord('i'):  # İstatistikler
                print(f"\nSLAM Istatistikleri:")
                print(f"   Toplam Nokta: {stats.get('mps', 0)}")
                print(f"   Keyframe: {stats.get('kfs', 0)}")
                print(f"   Kamera Pozisyonu: {stats.get('pos', (0,0,0))}")
            elif key >= ord('1') and key <= ord('6'):
                mode_manager.switch_mode(key - ord('0'))
                cv2.destroyAllWindows()
                speech_queue.put(f"MOD_{key - ord('0')}")
            continue
        
        # ============================================================
        # MOD 2: METİN OKUMA - Manuel tetikleme (Boşluk tuşu)
        # ============================================================
        if current_mode == 2:
            # OCR'ı ilk kullanımda yükle (lazy loading)
            if not ocr_reader.initialized:
                print("OCR sistemi yukleniyor (ilk kullanim)...")
                ocr_reader.init()
            
            # Sadece kamera görüntüsü göster
            display_frame = frame.copy()
            
            # MOD bilgisi ekle
            cv2.rectangle(display_frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, "MOD 2: METIN OKUMA [Bosluk=Oku]", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Son okunan metni göster
            if hasattr(mode_manager, 'last_ocr_text') and mode_manager.last_ocr_text:
                # Arka plan
                cv2.rectangle(display_frame, (10, 60), (630, 120), (0, 100, 0), -1)
                # Metin (kısa göster)
                short_text = mode_manager.last_ocr_text[:60] + "..." if len(mode_manager.last_ocr_text) > 60 else mode_manager.last_ocr_text
                cv2.putText(display_frame, f"Okunan: {short_text}", (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Sadece tek pencere göster
            cv2.imshow("Metin Okuma - MOD 2", display_frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # BOŞLUK TUŞU: OCR OKUMA
            elif key == ord(' '):
                print("\n" + "="*50)
                print("OCR TARAMASI BASLIYOR...")
                print("="*50)
                
                # OCR çalıştır
                text = read_text(frame.copy())
                
                if text:
                    # Aynı metin mi kontrol et (tekrar okumayı engelle)
                    prev_text = getattr(mode_manager, 'last_ocr_text', '')
                    
                    # Metinler çok benziyorsa (ilk 50 karakter aynıysa) tekrar okuma
                    if prev_text and text[:50] == prev_text[:50]:
                        print("[INFO] Ayni metin - tekrar okunmuyor")
                    else:
                        mode_manager.last_ocr_text = text
                        print(f"[OK] METIN BULUNDU: {text}")
                        print("Seslendiriliyor...")
                        speak_text_async(text)
                else:
                    mode_manager.last_ocr_text = "(Metin bulunamadı)"
                    print("[HATA] Metin bulunamadi")
                    speak_text_async("Metin bulunamadı")
                
                print("="*50 + "\n")
            
            elif key == ord('1'):
                mode_manager.switch_mode(1)
                cv2.destroyAllWindows()
                speech_queue.put("MOD_1")
                print(f"\n{'='*40}\nMOD 1: NAVIGASYON MODU AKTIF\n{'='*40}")
            elif key == ord('3'):
                mode_manager.switch_mode(3)
                cv2.destroyAllWindows()
                speech_queue.put("MOD_3")
                print(f"\n{'='*40}\nMOD 3: NESNE TANIMA MODU AKTIF\n{'='*40}")
            elif key == ord('4'):
                mode_manager.switch_mode(4)
                cv2.destroyAllWindows()
                speech_queue.put("MOD_4")
                search_target = input("Aranacak nesne: ").strip()
                print(f"\n{'='*40}\nMOD 4: NESNE ARAMA MODU AKTIF\n{'='*40}")
            elif key == ord('5'):
                # MOD 5'e geçiş (Sesli Sohbet) - ayrı döngü gerektirir
                print("[UYARI] MOD 5 icin programi yeniden baslatin")
            elif key == ord('6'):
                mode_manager.switch_mode(6)
                cv2.destroyAllWindows()
                speech_queue.put("MOD_6")
                print(f"\n{'='*40}\nMOD 6: GORSEL SORU-CEVAP MODU AKTIF\n{'='*40}")
            elif key == ord('7'):
                mode_manager.switch_mode(7)
                cv2.destroyAllWindows()
                slam_mapper.init()
                speech_queue.put("MOD_7")
                print(f"\n{'='*40}\nMOD 7: 3D HARITALAMA MODU AKTIF\n{'='*40}")
            continue
        
        # ============================================================
        # MOD 1, 3, 4 için YOLO ve Vision Pipeline çalışır
        # ============================================================
        
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
            
            # Bounding box'ı frame sınırları içinde tut
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            obstacles.append((x1, y1, x2, y2))
            
            # Mesafe tahmini
            bbox_height = y2 - y1
            distance, dist_category = estimate_distance(y2, frame_height, bbox_height)
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
        
        # === RADAR NAVİGASYON SİSTEMİ (YÖN KOMUTU KAYNAĞI) ===
        # YOLO engellerini radar'a aktar (mesafe bilgisi ile)
        radar_obstacles = []
        for item in pipeline_obstacles:
            x1, y1, x2, y2, class_name, confidence = item
            bbox_height = y2 - y1
            dist, _ = estimate_distance(y2, frame_height, bbox_height)
            radar_obstacles.append((x1, y1, x2, y2, class_name, dist))
        
        # Radar'ı güncelle - YÖN KOMUTU RADAR'DAN GELİYOR
        radar_img, radar_direction, radar_info = radar.process_frame(
            radar_obstacles, 
            frame_width, 
            frame_height
        )
        
        # RADAR YÖN KOMUTUNU ANA YÖN OLARAK KULLAN
        raw_direction = radar_direction  # Radar yönü ana yön kaynağı
        direction = stabilize_direction(raw_direction)  # Stabil yön
        
        # Cooldown azalt
        if speech_cooldown > 0:
            speech_cooldown -= 1
        if danger_cooldown > 0:
            danger_cooldown -= 1
        
        # === AKILLI GERÇEK ZAMANLI YÖNLENDİRME SİSTEMİ ===
        # SmartNavigator kullanarak akıcı ve tutarlı komutlar
        
        # Ortada engel var mı kontrol et
        orta_engel_var = False
        engel_bolge = None  # "SOL", "ORTA", "SAG"
        if closest_bbox is not None:
            cx = (closest_bbox[0] + closest_bbox[2]) // 2
            left_end = frame_width // 3
            right_start = 2 * frame_width // 3
            
            if cx < left_end:
                engel_bolge = "SOL"
            elif cx > right_start:
                engel_bolge = "SAG"
            else:
                engel_bolge = "ORTA"
                orta_engel_var = True
        
        # Radar yönünü akıllı navigatöre ekle
        smart_nav.add_direction(direction)
        
        # === KAÇIŞ YÖNÜ HESAPLA ===
        # Radar'ın önerdiği güvenli yön
        kacis_yonu = None
        if direction in ["SOL", "HAFIF_SOL"]:
            kacis_yonu = "SOL"
        elif direction in ["SAG", "HAFIF_SAG"]:
            kacis_yonu = "SAG"
        elif direction == "DÜZ":
            kacis_yonu = None  # Düz gidebilir
        
        # ACİL DURUMLAR - Kaçış yönü ile birlikte
        is_emergency = False
        emergency_command = None
        
        # 1. ÇOK YAKIN ENGEL - Kaçış yönü ile
        if closest_category == "COK_YAKIN" and orta_engel_var:
            is_emergency = True
            if kacis_yonu == "SOL":
                emergency_command = "DUR_SOL"
            elif kacis_yonu == "SAG":
                emergency_command = "DUR_SAG"
            else:
                emergency_command = "DUR_GERI"
        
        # 2. DUR komutu - Kaçış yönü ile
        elif direction == "DUR":
            is_emergency = True
            # Engel neredeyse oradan kaç
            if engel_bolge == "SOL":
                emergency_command = "ENGEL_SOL"  # Engel solda, sağa git
            elif engel_bolge == "SAG":
                emergency_command = "ENGEL_SAG"  # Engel sağda, sola git
            else:
                emergency_command = "DUR_GERI"
        
        # 3. YAKIN engel uyarısı - Kaçış yönü ile
        elif closest_category == "YAKIN" and orta_engel_var and min_distance and min_distance < 2.0:
            is_emergency = True
            if kacis_yonu == "SOL":
                emergency_command = "YAKIN_SOL"
            elif kacis_yonu == "SAG":
                emergency_command = "YAKIN_SAG"
            else:
                emergency_command = "YAKIN"
        
        # Acil durum varsa hemen söyle (kaçış yönü ile) - SADECE NAVİGASYON MODUNDA
        if is_emergency and danger_cooldown <= 0 and mode_manager.current_mode == 1:
            while not speech_queue.empty():
                try: speech_queue.get_nowait()
                except: pass
            speech_queue.put(emergency_command)
            danger_cooldown = 40  # 1.3 saniye - daha hızlı tepki
            speech_cooldown = 40
            smart_nav.last_command = emergency_command
            smart_nav.last_command_time = frame_count
            print(f"🚨 ACİL: {emergency_command}")
        
        # Normal yönlendirme - Akıllı navigator karar verir - SADECE NAVİGASYON MODUNDA
        elif not is_emergency and mode_manager.current_mode == 1:
            speak_command = smart_nav.update_state(direction, frame_count)
            
            if speak_command and speech_cooldown <= 0:
                while not speech_queue.empty():
                    try: speech_queue.get_nowait()
                    except: pass
                speech_queue.put(speak_command)
                last_spoken_direction = speak_command
                speech_cooldown = 30  # 1 saniye
                print(f"🎯 YÖN: {speak_command}")
        
        # Cooldown azalt
        if speech_cooldown > 0:
            speech_cooldown -= 1
        
        # Görselleştirme
        combined_view = draw_regions(combined_view, direction)
        
        # MOD BİLGİSİNİ EKRANA EKLE
        mode_text = f"MOD {mode_manager.current_mode}: {mode_manager.get_mode_name()}"
        cv2.rectangle(combined_view, (frame_width - 220, 5), (frame_width - 5, 35), (0, 0, 0), -1)
        cv2.putText(combined_view, mode_text, (frame_width - 215, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Bilgi göster
        if min_distance is not None:
            dist_color = (0, 0, 255) if closest_category in ["YAKIN", "COK_YAKIN"] else (0, 255, 255) if closest_category == "ORTA" else (0, 255, 0)
            cv2.putText(combined_view, f"Mesafe: {min_distance:.1f}m", (10, frame_height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
        
        cv2.putText(combined_view, f"Engel: {len(obstacles)}", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === MOD'A GÖRE İŞLEM YAP ===
        current_mode = mode_manager.current_mode
        
        # MOD 1: NAVİGASYON (varsayılan davranış - yukarıda yapıldı)
        # Zaten yön komutları speech sisteminde işleniyor
        
        # MOD 2: METİN OKUMA - YUKARIDAKİ ÖZEL BLOKTA İŞLENİYOR
        # (continue ile atlanıyor, buraya ulaşmaz)
        
        # MOD 3: NESNE TANIMA (services/object_describer.py kullanır)
        if current_mode == 3:
            if frame_count - last_describe_time > 90:  # Her 3 saniyede
                last_describe_time = frame_count
                description = describe_objects(pipeline_obstacles, frame_width, frame_height)
                print(f"[NESNE] {description}")
                speak_text_async(description)
        
        # MOD 4: NESNE ARAMA (services/object_searcher.py kullanır)
        elif current_mode == 4:
            if search_target and frame_count - last_search_time > 60:  # Her 2 saniyede
                last_search_time = frame_count
                
                # pipeline_obstacles'a mesafe bilgisi ekle
                obstacles_with_distance = []
                for item in pipeline_obstacles:
                    x1, y1, x2, y2, class_name, confidence = item
                    bbox_height = y2 - y1
                    distance, _ = estimate_distance(y2, frame_height, bbox_height)
                    obstacles_with_distance.append((x1, y1, x2, y2, class_name, confidence, distance))
                
                # Artık her zaman sonuç döner (bulunamadı dahil)
                result = search_object(obstacles_with_distance, search_target, frame_width, frame_height, report_not_found=True)
                if result:
                    print(f"[ARAMA] {result}")
                    speak_text_async(result)
            
            # Arama hedefini ekranda göster
            if search_target:
                cv2.putText(combined_view, f"Araniyor: {search_target}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # MOD 6: GÖRSEL SORU-CEVAP (services/image_qa.py kullanır)
        elif current_mode == 6:
            # Modül hazır mı kontrol et (lazy loading)
            if not image_qa.is_ready():
                if not image_qa.init():
                    print("[HATA] Gorsel soru-cevap baslatilamadi!")
                    speak_text_async("Görsel soru cevap başlatılamadı")
                    mode_manager.switch_mode(1)  # Navigasyona dön
                    current_mode = 1
                else:
                    speak_text_async("Görsel soru cevap hazır. Sorunuzu sormak için boşluk tuşuna basın.")
            
            # Ekranda bilgi göster
            cv2.putText(combined_view, "GORSEL SORU-CEVAP", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(combined_view, "Bosluk: Soru yaz (terminale)", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Debug (her 30 karede) - SADECE NAVİGASYON MODUNDA
        if frame_count % 30 == 0 and current_mode == 1:
            dist_str = f"{min_distance:.1f}m" if min_distance else "Yok"
            print(f"[MOD {current_mode}] Kare: {frame_count} | Engel: {len(obstacles)} | Yon: {direction}")
        
        # ============================================================
        # MOD'A GÖRE PENCERE GÖSTER
        # ============================================================
        if current_mode == 1:
            # NAVİGASYON: Tüm pencereler
            cv2.imshow("Navigasyon - MOD 1", combined_view)
            cv2.imshow("Kus Bakisi (BEV)", bev_combined)
            cv2.imshow("RADAR Navigasyon", radar_img)
        elif current_mode == 3:
            # NESNE TANIMA: Sadece ana görüntü
            cv2.imshow("Nesne Tanima - MOD 3", combined_view)
        elif current_mode == 4:
            # NESNE ARAMA: Sadece ana görüntü
            cv2.imshow("Nesne Arama - MOD 4", combined_view)
        elif current_mode == 6:
            # GÖRSEL SORU-CEVAP: Sadece ana görüntü
            cv2.imshow("Gorsel Soru-Cevap - MOD 6", combined_view)
        
        # TUŞ KONTROLLERI
        key = cv2.waitKey(1) & 0xFF
        
        # Çıkış
        if key == ord('q'):
            print("\nProgram sonlandiriliyor...")
            break
        
        # MOD 6: BOŞLUK TUŞU İLE SORU SOR (YAZILI)
        elif key == ord(' ') and mode_manager.current_mode == 6:
            if image_qa.is_ready():
                print("\n" + "=" * 50)
                print("Fotograf cekiliyor...")
                
                # Mevcut frame'i kaydet
                current_frame = frame.copy()
                
                # Kullanıcıdan yazılı soru al
                print("Sorunuzu yazin:")
                question = input("Soru: ").strip()
                
                if question:
                    print(f"[SORU] {question}")
                    print("Gemini analiz ediyor...")
                    speak_text_async("Analiz ediyorum, lütfen bekleyin")
                    
                    # Gemini'ye gönder
                    answer = image_qa.process_query(current_frame, question)
                    
                    print(f"[YANIT] {answer}")
                    speak_text_async(answer)
                else:
                    print("[HATA] Soru girilmedi")
                
                print("=" * 50 + "\n")
        
        # MOD DEĞİŞTİRME
        elif key == ord('1'):
            mode_manager.switch_mode(1)
            cv2.destroyAllWindows()
            print(f"\n{'='*40}")
            print(f"MOD 1: NAVIGASYON MODU AKTIF")
            print(f"{'='*40}")
            speech_queue.put("MOD_1")
        
        elif key == ord('2'):
            mode_manager.switch_mode(2)
            cv2.destroyAllWindows()
            print(f"\n{'='*40}")
            print(f"MOD 2: METIN OKUMA MODU AKTIF")
            print(f"{'='*40}")
            speech_queue.put("MOD_2")
        
        elif key == ord('3'):
            mode_manager.switch_mode(3)
            cv2.destroyAllWindows()
            print(f"\n{'='*40}")
            print(f"MOD 3: NESNE TANIMA MODU AKTIF")
            print(f"{'='*40}")
            speech_queue.put("MOD_3")
        
        elif key == ord('4'):
            mode_manager.switch_mode(4)
            print(f"\n{'='*40}")
            print(f"MOD 4: NESNE ARAMA MODU AKTIF")
            print("Aramak istediğiniz nesneyi yazın (örn: insan, sandalye, telefon):")
            print(f"{'='*40}")
            speech_queue.put("MOD_4")
            # Terminalde arama hedefi al
            search_target = input("Aranacak nesne: ").strip()
            if search_target:
                print(f"[ARAMA] '{search_target}' araniyor...")
                speak_text_async(f"{search_target} aranıyor")
                search_found = False
        
        # MOD 4'te yeni arama
        elif key == ord('s') and mode_manager.current_mode == 4:
            print("Yeni arama hedefi girin:")
            search_target = input("Aranacak nesne: ").strip()
            if search_target:
                print(f"[ARAMA] '{search_target}' araniyor...")
                speak_text_async(f"{search_target} aranıyor")
                search_found = False
        
        elif key == ord('6'):
            mode_manager.switch_mode(6)
            cv2.destroyAllWindows()
            print(f"\n{'='*40}")
            print(f"MOD 6: GORSEL SORU-CEVAP MODU AKTIF")
            print(f"   Boşluk tuşuna basarak soru sorun")
            print(f"{'='*40}")
            speech_queue.put("MOD_6")
        
        elif key == ord('7'):
            mode_manager.switch_mode(7)
            cv2.destroyAllWindows()
            slam_mapper.init()
            print(f"\n{'='*40}")
            print(f"MOD 7: 3D HARITALAMA MODU AKTIF")
            print(f"   SPACE: Kaydet | L: Yükle | R: Sıfırla")
            print(f"{'='*40}")
            speech_queue.put("MOD_7")
    
    # Temizlik
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    
    # MOD 6 temizliği
    if image_qa.is_ready():
        image_qa.cleanup()
    
    print("Program basariyla sonlandirildi.")


def main_voice_controlled():
    """
    SESLİ KONTROLLÜ ANA FONKSİYON
    Tüm modlar sesli komutlarla kontrol edilir.
    
    MOD SEÇİMİ:
      - "navigasyon" -> Mod 1
      - "metin" -> Mod 2
      - "tanıma" -> Mod 3
      - "arama" -> Mod 4
      - "sohbet" -> Mod 5
      - "soru" -> Mod 6
      - "harita" -> Mod 7
    
    GENEL KOMUTLAR:
      - "çık" -> Mod menüsüne dön
      - "kapat" -> Programı kapat
    
    MOD ÖZELLİKLERİ:
      - Mod 2: "çek" -> OCR okuma
      - Mod 4: "[nesne] ara" -> Nesne ara
      - Mod 5: Sesli sohbet (direkt konuşma)
      - Mod 6: "çek" -> Fotoğraf çek, sonra soru sor
    """
    global speech_thread_running, mode_manager
    
    print("=" * 60)
    print("    GÖRME ENGELLİ ASİSTANI - SESLİ KONTROL")
    print("=" * 60)
    print("\nSESLİ KOMUTLAR:")
    print("  'navigasyon' - Navigasyon Modu")
    print("  'metin'      - Metin Okuma Modu")
    print("  'tanıma'     - Nesne Tanıma Modu")
    print("  'arama'      - Nesne Arama Modu")
    print("  'sohbet'     - Sesli AI Sohbet Modu")
    print("  'soru'       - Görsel Soru-Cevap Modu")
    print("  'harita'     - 3D Haritalama Modu")
    print("  'kapat'      - Programı Kapat")
    print("-" * 60)
    
    # Ses dosyalarını oluştur
    print("\nSes dosyalari hazirlaniyor...")
    create_audio_files()
    
    # Ses thread'ini başlat
    print("Ses sistemi baslatiliyor...")
    speech_thread_running = True
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    # Sesli komut sistemini başlat
    print("Sesli komut sistemi baslatiliyor...")
    if not voice_command.init():
        print("[HATA] Sesli komut sistemi baslatilamadi!")
        print("  Mikrofon kontrol edin ve pyaudio yuklu oldugundan emin olun.")
        print("  Program klavye kontrolu ile devam edecek...")
        main()  # Fallback to keyboard control
        return
    
    print("[OK] Sesli komut sistemi hazir!")
    
    # Test sesi
    speak_text_async("Sistem hazır. Mod seçmek için komut verin.")
    time.sleep(2)
    
    # Kamerayı aç
    print("\nKamera açılıyor...")
    cap = None
    
    if PICAMERA_AVAILABLE:
        try:
            print("Pi Camera Module v3 deneniyor...")
            cap = PiCameraReader(camera_num=0, width=1280, height=720)
            print("[OK] Pi Camera basariyla baslatildi!")
        except Exception as e:
            print(f"[UYARI] Pi Camera baslatilamadi: {e}")
            cap = None
    
    if cap is None or not cap.isOpened():
        print("USB/Varsayılan kamera deneniyor...")
        cap = LatestFrameReader(0)
        if not cap.isOpened():
            print("[HATA] Kamera acilamadi!")
            return
    
    # Vision Pipeline
    print("Vision Pipeline baslatiliyor...")
    try:
        pipeline = VisionPipeline("../models/yolo11n.pt")
        print("[OK] Pipeline hazir!")
    except Exception as e:
        print(f"[HATA] Pipeline baslatma hatasi: {e}")
        cap.release()
        return
    
    # 3D Navigasyon Haritası
    nav_map = NavigationMap(grid_size=(80, 80), cell_size=0.1)
    depth_estimator = DepthEstimator()
    
    # RADAR NAVİGASYON SİSTEMİ
    radar = RadarNavigation(radar_size=400)
    print("[OK] Radar navigasyon sistemi hazir!")
    
    # Cooldown değişkenleri
    speech_cooldown = 0
    danger_cooldown = 0
    
    # ========== ANA DÖNGÜ ==========
    running = True
    current_mode = 0  # 0 = mod seçim menüsü
    search_target = None
    frame_count = 0
    
    print("\n" + "=" * 60)
    print("    MOD SEÇİM MENÜSÜ - SESLİ KOMUT BEKLENİYOR")
    print("=" * 60)
    speak_text_async("Mod seçin. Navigasyon, metin, tanıma, arama, sohbet, soru veya harita deyin.")
    
    while running:
        # Kamera frame al
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # ============================================================
        # MOD 0: MOD SEÇİM MENÜSÜ
        # ============================================================
        if current_mode == 0:
            # Görüntü göster
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (500, 60), (0, 0, 0), -1)
            cv2.putText(display_frame, "MOD SECIM MENUSU - Sesli komut verin", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Gorme Engelli Asistani", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # Sesli komut dinle (non-blocking şekilde)
            if frame_count % 30 == 0:  # Her 30 frame'de bir dinle
                cmd_type, cmd_value = voice_command.wait_for_mode()
                
                if cmd_type == 'shutdown':
                    speak_text_async("Program kapatılıyor. Hoşça kalın!")
                    time.sleep(2)
                    running = False
                    break
                
                elif cmd_type == 'mode':
                    current_mode = cmd_value
                    mode_manager.switch_mode(current_mode)
                    print(f"\n[OK] MOD {current_mode}: {mode_manager.get_mode_name()} AKTIF")
                    speech_queue.put(f"MOD_{current_mode}")
                    
                    # Mod 7 için SLAM başlat
                    if current_mode == 7:
                        slam_mapper.init()
                    
                    # Mod 5 için ayrı döngüye git
                    if current_mode == 5:
                        cv2.destroyAllWindows()
                        run_voice_chat_mode()
                        current_mode = 0  # Geri dön
                        speak_text_async("Mod menüsüne döndük. Yeni mod seçin.")
                        continue
                    
                    time.sleep(1)
            
            continue
        
        # ============================================================
        # MOD 1: NAVİGASYON (KLAVYE MODUYLA AYNI)
        # ============================================================
        elif current_mode == 1:
            height, width = frame.shape[:2]
            
            # Vision Pipeline ile işle (YOLO + Zemin Tespiti + IPM)
            combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
            
            # Free Space Overlay
            free_space_overlay = np.zeros_like(bev_view)
            free_space_overlay[free_space_mask > 0] = [0, 255, 0]
            bev_combined = cv2.addWeighted(bev_view, 0.7, free_space_overlay, 0.3, 0)
            
            # Engelleri topla
            obstacles = []
            for item in pipeline_obstacles:
                x1, y1, x2, y2, class_name, confidence = item
                
                # Bounding box'ı frame sınırları içinde tut
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                obstacles.append((x1, y1, x2, y2))
                
                # Mesafe tahmini
                bbox_height = y2 - y1
                distance, dist_category = estimate_distance(y2, height, bbox_height)
                if dist_category in ["YAKIN", "COK_YAKIN"]:
                    box_color = (0, 0, 255)  # Kırmızı
                elif dist_category == "ORTA":
                    box_color = (0, 165, 255)  # Turuncu
                else:
                    box_color = (0, 255, 0)  # Yeşil
                
                # Bounding box çiz
                cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
                
                # Label pozisyonu
                label = f"{class_name}: {distance:.1f}m"
                label_y = y1 - 5 if y1 > 20 else y2 + 15
                cv2.putText(combined_view, label, (x1, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # En yakın engeli bul
            min_distance, closest_category, closest_bbox = get_closest_obstacle(obstacles, height)
            
            # === RADAR NAVİGASYON SİSTEMİ ===
            radar_obstacles = []
            for item in pipeline_obstacles:
                x1, y1, x2, y2, class_name, confidence = item
                bbox_height = y2 - y1
                dist, _ = estimate_distance(y2, height, bbox_height)
                radar_obstacles.append((x1, y1, x2, y2, class_name, dist))
            
            # Radar'ı güncelle
            radar_img, radar_direction, radar_info = radar.process_frame(
                radar_obstacles, 
                width, 
                height
            )
            
            # RADAR YÖN KOMUTUNU KULLAN
            raw_direction = radar_direction
            direction = stabilize_direction(raw_direction)
            
            # Cooldown azalt
            if speech_cooldown > 0:
                speech_cooldown -= 1
            if danger_cooldown > 0:
                danger_cooldown -= 1
            
            # === AKILLI YÖNLENDİRME SİSTEMİ ===
            orta_engel_var = False
            engel_bolge = None
            if closest_bbox is not None:
                cx = (closest_bbox[0] + closest_bbox[2]) // 2
                left_end = width // 3
                right_start = 2 * width // 3
                
                if cx < left_end:
                    engel_bolge = "SOL"
                elif cx > right_start:
                    engel_bolge = "SAG"
                else:
                    engel_bolge = "ORTA"
                    orta_engel_var = True
            
            smart_nav.add_direction(direction)
            
            # Kaçış yönü hesapla
            kacis_yonu = None
            if direction in ["SOL", "HAFIF_SOL"]:
                kacis_yonu = "SOL"
            elif direction in ["SAG", "HAFIF_SAG"]:
                kacis_yonu = "SAG"
            
            # ACİL DURUMLAR
            is_emergency = False
            emergency_command = None
            
            if closest_category == "COK_YAKIN" and orta_engel_var:
                is_emergency = True
                if kacis_yonu == "SOL":
                    emergency_command = "DUR_SOL"
                elif kacis_yonu == "SAG":
                    emergency_command = "DUR_SAG"
                else:
                    emergency_command = "DUR_GERI"
            elif direction == "DUR":
                is_emergency = True
                if engel_bolge == "SOL":
                    emergency_command = "ENGEL_SOL"
                elif engel_bolge == "SAG":
                    emergency_command = "ENGEL_SAG"
                else:
                    emergency_command = "DUR_GERI"
            elif closest_category == "YAKIN" and orta_engel_var and min_distance and min_distance < 2.0:
                is_emergency = True
                if kacis_yonu == "SOL":
                    emergency_command = "YAKIN_SOL"
                elif kacis_yonu == "SAG":
                    emergency_command = "YAKIN_SAG"
                else:
                    emergency_command = "YAKIN"
            
            # Acil durum seslendirme
            if is_emergency and danger_cooldown <= 0:
                while not speech_queue.empty():
                    try: speech_queue.get_nowait()
                    except: pass
                speech_queue.put(emergency_command)
                danger_cooldown = 40
                speech_cooldown = 40
                smart_nav.last_command = emergency_command
                smart_nav.last_command_time = frame_count
                print(f"[ACIL] {emergency_command}")
            
            # Normal yönlendirme
            elif not is_emergency:
                speak_command = smart_nav.update_state(direction, frame_count)
                
                if speak_command and speech_cooldown <= 0:
                    while not speech_queue.empty():
                        try: speech_queue.get_nowait()
                        except: pass
                    speech_queue.put(speak_command)
                    speech_cooldown = 30
                    print(f"[YON] {speak_command}")
            
            # Görselleştirme
            combined_view = draw_regions(combined_view, direction)
            
            # MOD bilgisi ekle
            cv2.rectangle(combined_view, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(combined_view, "MOD 1: NAVIGASYON (SESLI)", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mesafe bilgisi
            if min_distance is not None:
                dist_color = (0, 0, 255) if closest_category in ["YAKIN", "COK_YAKIN"] else (0, 255, 255) if closest_category == "ORTA" else (0, 255, 0)
                cv2.putText(combined_view, f"Mesafe: {min_distance:.1f}m", (10, height - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
            
            cv2.putText(combined_view, f"Engel: {len(obstacles)}", (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # TÜM PENCERELERİ GÖSTER (KLAVYE MODUYLA AYNI)
            cv2.imshow("Navigasyon - MOD 1", combined_view)
            cv2.imshow("Kus Bakisi (BEV)", bev_combined)
            cv2.imshow("RADAR Navigasyon", radar_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # Sesli komut kontrolü
            if frame_count % 60 == 0:
                action, value = voice_command.wait_for_action(1)
                if action == 'exit':
                    current_mode = 0
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
        
        # ============================================================
        # MOD 2: METİN OKUMA
        # ============================================================
        elif current_mode == 2:
            if not ocr_reader.initialized:
                ocr_reader.init()
            
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, "MOD 2: METIN OKUMA [Cek=Oku]", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if hasattr(mode_manager, 'last_ocr_text') and mode_manager.last_ocr_text:
                cv2.rectangle(display_frame, (10, 60), (630, 120), (0, 100, 0), -1)
                short_text = mode_manager.last_ocr_text[:60] + "..." if len(mode_manager.last_ocr_text) > 60 else mode_manager.last_ocr_text
                cv2.putText(display_frame, f"Okunan: {short_text}", (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Metin Okuma - MOD 2", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            elif key == ord(' '):  # Klavye ile de çekilebilsin
                text = read_text(frame.copy())
                if text:
                    mode_manager.last_ocr_text = text
                    print(f"[OK] METIN: {text}")
                    speak_text_async(text)
                else:
                    speak_text_async("Metin bulunamadı")
            
            # Sesli komut kontrolü
            if frame_count % 45 == 0:
                action, value = voice_command.wait_for_action(2)
                if action == 'exit':
                    current_mode = 0
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
                elif action == 'capture':
                    print("[OCR] Çek komutu alındı...")
                    speak_text_async("Metin okunuyor")
                    text = read_text(frame.copy())
                    if text:
                        mode_manager.last_ocr_text = text
                        print(f"[OK] METIN: {text}")
                        speak_text_async(text)
                    else:
                        speak_text_async("Metin bulunamadı")
        
        # ============================================================
        # MOD 3: NESNE TANIMA (KLAVYE MODUYLA AYNI)
        # ============================================================
        elif current_mode == 3:
            height, width = frame.shape[:2]
            
            # Vision Pipeline ile işle
            combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
            
            # Engelleri işle ve çiz
            for item in pipeline_obstacles:
                x1, y1, x2, y2, class_name, confidence = item
                
                # Bounding box sınırları
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Mesafe tahmini
                bbox_height = y2 - y1
                distance, dist_category = estimate_distance(y2, height, bbox_height)
                
                # Renk (mesafeye göre)
                if dist_category in ["YAKIN", "COK_YAKIN"]:
                    box_color = (0, 0, 255)
                elif dist_category == "ORTA":
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 255, 0)
                
                # Türkçe isim
                turkish_label = get_turkish_name(class_name)
                
                # Bounding box çiz
                cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
                label = f"{turkish_label}: {distance:.1f}m"
                label_y = y1 - 5 if y1 > 20 else y2 + 15
                cv2.putText(combined_view, label, (x1, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Her 3 saniyede nesneleri seslendir (klavye modundaki gibi)
            if not hasattr(mode_manager, 'last_describe_time'):
                mode_manager.last_describe_time = 0
            
            if frame_count - mode_manager.last_describe_time > 90:
                mode_manager.last_describe_time = frame_count
                description = describe_objects(pipeline_obstacles, width, height)
                print(f"[NESNE] {description}")
                speak_text_async(description)
            
            # MOD bilgisi ekle
            cv2.rectangle(combined_view, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(combined_view, "MOD 3: NESNE TANIMA (SESLI)", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Nesne sayısı
            cv2.putText(combined_view, f"Tespit: {len(pipeline_obstacles)} nesne", (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Nesne Tanima - MOD 3", combined_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # Sesli komut kontrolü
            if frame_count % 60 == 0:
                action, value = voice_command.wait_for_action(3)
                if action == 'exit':
                    current_mode = 0
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
        
        # ============================================================
        # MOD 4: NESNE ARAMA (KLAVYE MODUYLA AYNI)
        # ============================================================
        elif current_mode == 4:
            height, width = frame.shape[:2]
            
            if search_target is None:
                # Hedef sor (sadece ilk seferde)
                if not hasattr(mode_manager, 'search_prompt_shown') or not mode_manager.search_prompt_shown:
                    speak_text_async("Aramak istediğiniz nesneyi söyleyin, sonra ara deyin.")
                    mode_manager.search_prompt_shown = True
            
            # Vision Pipeline ile işle
            combined_view, pipeline_obstacles, edges, bev_view, free_space_mask = pipeline.process_frame(frame)
            
            # Engelleri işle ve çiz
            for item in pipeline_obstacles:
                x1, y1, x2, y2, class_name, confidence = item
                
                # Bounding box sınırları
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Mesafe tahmini
                bbox_height = y2 - y1
                distance, dist_category = estimate_distance(y2, height, bbox_height)
                
                # Aranan nesne mi kontrol et
                is_target = False
                if search_target:
                    if search_target.lower() in class_name.lower() or class_name.lower() in search_target.lower():
                        is_target = True
                
                # Renk (aranan nesne kırmızı, diğerleri yeşil)
                if is_target:
                    box_color = (0, 0, 255)  # Kırmızı - bulundu!
                    cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(combined_view, f"BULUNDU: {class_name} ({distance:.1f}m)", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                else:
                    box_color = (0, 255, 0)
                    cv2.rectangle(combined_view, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(combined_view, f"{class_name}: {distance:.1f}m", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Arama sonuçlarını seslendir (klavye modundaki gibi)
            if not hasattr(mode_manager, 'last_search_time'):
                mode_manager.last_search_time = 0
            
            if search_target and frame_count - mode_manager.last_search_time > 60:
                mode_manager.last_search_time = frame_count
                
                # pipeline_obstacles'a mesafe bilgisi ekle
                obstacles_with_distance = []
                for item in pipeline_obstacles:
                    x1, y1, x2, y2, class_name, confidence = item
                    bbox_height = y2 - y1
                    dist, _ = estimate_distance(y2, height, bbox_height)
                    obstacles_with_distance.append((x1, y1, x2, y2, class_name, confidence, dist))
                
                # Artık her zaman sonuç döner (bulunamadı dahil)
                result = search_object(obstacles_with_distance, search_target, width, height, report_not_found=True)
                if result:
                    print(f"[ARAMA] {result}")
                    speak_text_async(result)
            
            # MOD bilgisi ekle
            cv2.rectangle(combined_view, (10, 10), (450, 50), (0, 0, 0), -1)
            target_text = search_target if search_target else "Hedef yok"
            cv2.putText(combined_view, f"MOD 4: ARAMA - {target_text}", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Arama hedefini ekranda göster
            if search_target:
                cv2.putText(combined_view, f"Araniyor: {search_target}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Nesne Arama - MOD 4", combined_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # Sesli komut kontrolü
            if frame_count % 45 == 0:
                action, value = voice_command.wait_for_action(4)
                if action == 'exit':
                    current_mode = 0
                    search_target = None
                    mode_manager.search_prompt_shown = False
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
                elif action == 'search':
                    search_target = value
                    print(f"[ARAMA] Yeni hedef: {search_target}")
                    speak_text_async(f"{search_target} aranıyor")
                elif action == 'set_target':
                    search_target = value
                    print(f"[ARAMA] Hedef ayarlandı: {search_target}")
        
        # ============================================================
        # MOD 6: GÖRSEL SORU-CEVAP (KLAVYE MODUYLA AYNI)
        # ============================================================
        elif current_mode == 6:
            # Modül hazır mı kontrol et (lazy loading)
            if not image_qa.is_ready():
                if not image_qa.init():
                    print("[HATA] Gorsel soru-cevap baslatilamadi!")
                    speak_text_async("Görsel soru cevap başlatılamadı")
                    current_mode = 0
                    continue
                else:
                    speak_text_async("Görsel soru cevap hazır. Boşluk tuşuna basın veya çek deyin.")
            
            display_frame = frame.copy()
            
            # MOD bilgisi ekle
            cv2.rectangle(display_frame, (10, 10), (500, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, "MOD 6: GORSEL SORU-CEVAP (SESLI)", (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            
            # Kullanım bilgisi
            cv2.putText(display_frame, "Bosluk: Yazili soru | 'cek': Sesli soru", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Gorsel Soru-Cevap - MOD 6", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # BOŞLUK TUŞU: YAZILI SORU SOR (klavye modundaki gibi)
            elif key == ord(' '):
                if image_qa.is_ready():
                    print("\n" + "=" * 50)
                    print("Fotograf cekiliyor...")
                    
                    # Mevcut frame'i kaydet
                    current_frame = frame.copy()
                    
                    # Terminalde soru al
                    print("Sorunuzu yazin:")
                    question = input("SORU: ").strip()
                    
                    if question:
                        print("Dusunuluyor...")
                        speak_text_async("Düşünüyorum")
                        
                        answer = image_qa.process_query(current_frame, question)
                        print(f"\n[CEVAP] {answer}")
                        print("=" * 50 + "\n")
                        speak_text_async(answer)
                    else:
                        print("Soru girilmedi.")
            
            # Sesli komut kontrolü
            if frame_count % 45 == 0:
                action, value = voice_command.wait_for_action(6)
                if action == 'exit':
                    current_mode = 0
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
                elif action == 'capture':
                    speak_text_async("Fotoğraf çekildi. Sorunuzu sorun.")
                    captured_frame = frame.copy()
                    
                    # Soru dinle
                    time.sleep(1)
                    success, question = voice_command.listen(timeout=10, phrase_limit=20)
                    if success and question:
                        print(f"[SORU] {question}")
                        speak_text_async("Düşünüyorum")
                        
                        # Gemini'ye sor
                        if not image_qa.is_ready():
                            image_qa.init()
                        answer = image_qa.process_query(captured_frame, question)
                        print(f"[CEVAP] {answer}")
                        speak_text_async(answer)
                elif action == 'speech':
                    # Direkt soru sorulmuş
                    question = value
                    speak_text_async("Düşünüyorum")
                    if not image_qa.is_ready():
                        image_qa.init()
                    answer = image_qa.process_query(frame.copy(), question)
                    print(f"[CEVAP] {answer}")
                    speak_text_async(answer)
        
        # ============================================================
        # MOD 7: 3D HARİTALAMA (KLAVYE MODUYLA AYNI)
        # ============================================================
        elif current_mode == 7:
            # SLAM frame işle
            success = slam_mapper.process_frame(frame)
            
            # SLAM görselleştirmesi al (frame üzerine çizim)
            slam_vis = slam_mapper.get_visualization(frame)
            
            # Kuş bakışı harita al
            topdown = slam_mapper.get_topdown_map()
            
            # İstatistikleri al
            stats = slam_mapper.get_stats()
            
            # MOD bilgisi ekle
            cv2.rectangle(slam_vis, (10, 10), (450, 80), (0, 0, 0), -1)
            cv2.putText(slam_vis, f"MOD 7: 3D HARITA (SESLI)", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(slam_vis, f"Noktalar: {stats.get('mps', 0)} | Keyframe: {stats.get('kfs', 0)}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Pencereleri göster (klavye modundaki gibi 2 pencere)
            cv2.imshow("SLAM Kamera", slam_vis)
            cv2.imshow("SLAM Harita", topdown)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # BOŞLUK TUŞU: Haritayı kaydet (klavye modundaki gibi)
            elif key == ord(' '):
                maps_dir = os.path.join(os.path.dirname(__file__), "maps")
                os.makedirs(maps_dir, exist_ok=True)
                filepath = os.path.join(maps_dir, f"room_map_{int(time.time())}.ply")
                if slam_mapper.save_map(filepath):
                    print(f"[OK] Harita kaydedildi: {filepath}")
                    speak_text_async("Harita kaydedildi")
                else:
                    print("[HATA] Harita kaydedilemedi (yeterli nokta yok)")
                    speak_text_async("Harita kaydedilemedi")
            
            # L TUŞU: Harita yükle
            elif key == ord('l'):
                maps_dir = os.path.join(os.path.dirname(__file__), "maps")
                if os.path.exists(maps_dir):
                    ply_files = [f for f in os.listdir(maps_dir) if f.endswith('.ply')]
                    if ply_files:
                        latest = sorted(ply_files)[-1]
                        filepath = os.path.join(maps_dir, latest)
                        if slam_mapper.load_map(filepath):
                            print(f"[OK] Harita yuklendi: {latest}")
                            speak_text_async("Harita yüklendi")
                        else:
                            print("[HATA] Harita yuklenemedi")
                    else:
                        print("[HATA] Kayitli harita bulunamadi")
                        speak_text_async("Kayıtlı harita yok")
            
            # R TUŞU: Haritayı sıfırla
            elif key == ord('r'):
                slam_mapper.reset()
                print("SLAM sifirlanadi")
                speak_text_async("Harita sıfırlandı")
            
            # I TUŞU: İstatistikler
            elif key == ord('i'):
                print(f"\nSLAM Istatistikleri:")
                print(f"   Toplam Nokta: {stats.get('mps', 0)}")
                print(f"   Keyframe: {stats.get('kfs', 0)}")
                print(f"   Kamera Pozisyonu: {stats.get('pos', (0,0,0))}")
            
            # Sesli komut kontrolü
            if frame_count % 60 == 0:
                action, value = voice_command.wait_for_action(7)
                if action == 'exit':
                    current_mode = 0
                    cv2.destroyAllWindows()
                    speak_text_async("Mod menüsüne döndük.")
                elif action == 'shutdown':
                    running = False
    
    # Temizlik
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    
    if image_qa.is_ready():
        image_qa.cleanup()
    
    print("\nProgram basariyla sonlandirildi.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--keyboard":
        # Klavye kontrollü eski mod
        main()
    else:
        # Sesli kontrollü yeni mod (varsayılan)
        main_voice_controlled()
