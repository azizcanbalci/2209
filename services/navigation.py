import cv2
import time
import os
from .base import BaseService
from .vision_pipeline import VisionPipeline

class NavigationService(BaseService):
    def __init__(self, audio_service):
        super().__init__("NavigationService")
        self.audio_service = audio_service
        self.pipeline = None
        self.cap = None
        
        # Model yolu - CWD'ye göre (root'tan çalıştırıldığında)
        self.model_path = "models/yolo11n.pt"
        if not os.path.exists(self.model_path):
            # Belki bir üst klasördedir (geliştirme ortamına göre)
            self.model_path = "../models/yolo11n.pt"

    def run(self):
        print("[NavigationService] Başlatılıyor...")
        
        # Pipeline başlat
        if self.pipeline is None:
            try:
                self.pipeline = VisionPipeline(self.model_path)
            except Exception as e:
                print(f"Pipeline hatası: {e}")
                self.audio_service.play("HATA")
                return

        # Kamera başlat
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            self.audio_service.play("HATA")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.audio_service.play("NAV_BASLA")
        
        speech_cooldown = 0
        danger_cooldown = 0
        frame_count = 0
        last_obstacles = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_height, frame_width = frame.shape[:2]
            frame_count += 1
            
            # 5 karede 1 YOLO çalıştır
            run_yolo = (frame_count % 5 == 0)
            
            # Pipeline işlemi
            try:
                # Yeni Pipeline Çıktısı: combined_view, obstacles, bev_edges, free_space_mask, direction
                combined_view, obstacles, bev_edges, free_space_mask, direction = self.pipeline.process_frame(frame, run_yolo=run_yolo)
                
                # Engelleri güncelle veya eskileri kullan
                if run_yolo:
                    last_obstacles = obstacles
                else:
                    obstacles = last_obstacles
                
                # En yakın engel kontrolü
                min_dist = float('inf')
                closest_cat = "UZAK"
                
                for obs in obstacles:
                    # obs: (x1, y1, x2, y2, label, conf)
                    x1, y1, x2, y2 = obs[:4]
                    h = y2 - y1
                    ratio = h / frame_height
                    
                    # Merkez kontrolü: Engel gerçekten önümüzde mi?
                    # Ekranın ortasındaki %60'lık dilime giriyor mu?
                    obs_center_x = (x1 + x2) / 2
                    screen_center = frame_width / 2
                    safe_zone_width = frame_width * 0.6 # Ortadaki %60
                    
                    in_center_zone = (screen_center - safe_zone_width/2) < obs_center_x < (screen_center + safe_zone_width/2)
                    
                    if not in_center_zone:
                        continue # Kenardaki engeller için "Çok Yakın" uyarısı verme
                    
                    # Mesafe Hesabı (Perspektif Tabanlı - Y Koordinatı)
                    # Nesnenin alt noktası (y2) ne kadar aşağıdaysa o kadar yakındır.
                    # 0.0 (Üst/Ufuk) -> 1.0 (Alt/Ayak ucu)
                    proximity = y2 / frame_height
                    
                    dist = 6.0
                    cat = "UZAK"
                    
                    # Eşik değerleri (Perspektife göre ayarlandı)
                    if proximity > 0.90: cat = "YAKIN"; dist = 0.5   # Ayak ucunda (Çok Tehlikeli)
                    elif proximity > 0.75: cat = "YAKIN"; dist = 1.0 # 1-2 metre (Tehlikeli)
                    elif proximity > 0.60: cat = "ORTA"; dist = 2.0  # 3-4 metre (Dikkat)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_cat = cat

                # Sesli uyarı mantığı
                if danger_cooldown > 0: danger_cooldown -= 1
                if speech_cooldown > 0: speech_cooldown -= 1
                
                if closest_cat == "YAKIN" and danger_cooldown <= 0:
                    self.audio_service.play_immediate("YAKIN")
                    danger_cooldown = 60
                elif speech_cooldown <= 0:
                    if direction != "DUR": # Sadece hareket varsa veya yön değiştiyse
                        self.audio_service.play(direction)
                        speech_cooldown = 90 # 3 saniye
                
                # Görselleştirme
                cv2.imshow("Navigasyon (Canny+IPM)", combined_view)
                cv2.imshow("BEV (Edges)", bev_edges)
                cv2.imshow("Free Space", free_space_mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except Exception as e:
                print(f"Döngü hatası: {e}")
                
        # Temizlik
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.audio_service.play("NAV_DUR")
