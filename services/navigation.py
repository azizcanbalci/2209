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
        last_direction = "DUR"
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_height, frame_width = frame.shape[:2]
            
            # Pipeline işlemi
            try:
                combined_view, obstacles, edges, bev_view, free_space_mask = self.pipeline.process_frame(frame)
                
                # Yön bulma
                direction = self.pipeline.find_best_direction(free_space_mask)
                
                # En yakın engel kontrolü (Basit mantık)
                # Not: VisionPipeline içinde mesafe hesabı yok, main.py'de vardı.
                # Buraya basit bir mesafe kontrolü ekleyelim veya pipeline'a taşıyalım.
                # Şimdilik pipeline çıktısındaki obstacles üzerinden gidelim.
                
                min_dist = float('inf')
                closest_cat = "UZAK"
                
                for obs in obstacles:
                    # obs: (x1, y1, x2, y2, label, conf)
                    x1, y1, x2, y2 = obs[:4]
                    h = y2 - y1
                    ratio = h / frame_height
                    
                    dist = 6.0
                    cat = "UZAK"
                    if ratio > 0.5: cat = "YAKIN"; dist = 0.5
                    elif ratio > 0.35: cat = "YAKIN"; dist = 1.0
                    elif ratio > 0.25: cat = "ORTA"; dist = 1.5
                    
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
                
                # Görselleştirme (Opsiyonel - Headless modda kapatılabilir)
                cv2.imshow("Navigasyon", combined_view)
                cv2.imshow("BEV", bev_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except Exception as e:
                print(f"Döngü hatası: {e}")
                
        # Temizlik
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.audio_service.play("NAV_DUR")
