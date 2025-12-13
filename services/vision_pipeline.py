import cv2
import numpy as np
from ultralytics import YOLO
import os

class VisionPipeline:
    def __init__(self, model_path="../models/yolo11n.pt"):
        """
        Görüntü işleme pipeline'ını başlatır.
        
        Args:
            model_path: YOLO model dosyasının yolu
        """
        print(f"VisionPipeline başlatılıyor... Model: {model_path}")
        
        # 1. YOLO Modelini Yükle
        try:
            self.model = YOLO(model_path)
            print("YOLO modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"YOLO yükleme hatası: {e}")
            raise e
            
        # IPM Matrisleri (Lazy initialization)
        self.ipm_matrix = None
        self.ipm_width = 0
        self.ipm_height = 0
        
        # Son tespit edilen engelleri sakla (YOLO her karede çalışmadığı için)
        self.last_obstacles = []

    def init_ipm(self, width, height):
        """
        IPM (Inverse Perspective Mapping) matrisini hesaplar.
        Kamera açısına göre bu noktaların kalibre edilmesi gerekebilir.
        """
        self.ipm_width = width
        self.ipm_height = height
        
        # Kaynak noktalar (Trapezoid - Kamera görüntüsündeki yol alanı)
        # Bu değerler 640x480 çözünürlük için yaklaşık değerlerdir
        # Alt kısım geniş, üst kısım dar
        src_points = np.float32([
            [width * 0.25, height * 0.55],  # Sol Üst
            [width * 0.75, height * 0.55],  # Sağ Üst
            [width * 0.05, height * 0.95],  # Sol Alt
            [width * 0.95, height * 0.95]   # Sağ Alt
        ])
        
        # Hedef noktalar (Dikdörtgen - Kuş bakışı görünüm)
        dst_points = np.float32([
            [0, 0],             # Sol Üst
            [width, 0],         # Sağ Üst
            [0, height],        # Sol Alt
            [width, height]     # Sağ Alt
        ])
        
        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print("IPM Matrisi hesaplandı.")

    def apply_ipm(self, image):
        """
        Görüntüye IPM dönüşümü uygular.
        """
        if self.ipm_matrix is None:
            h, w = image.shape[:2]
            self.init_ipm(w, h)
            
        return cv2.warpPerspective(image, self.ipm_matrix, (self.ipm_width, self.ipm_height))

    def detect_edges(self, frame, low_threshold=50, high_threshold=150):
        """
        Canny Kenar Tespiti (Multi-scale yaklaşımı basitleştirilmiş).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

    def apply_morphology(self, edges):
        """
        Morfolojik işlemler ile kenarları temizler ve birleştirir.
        """
        # Kenarları kalınlaştır (Dilation) - Kopuk kenarları birleştir
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Closing (Dilation + Erosion) - Küçük delikleri kapat
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        return closed

    def apply_pseudo_depth(self, edges):
        """
        "Sahte 3D" (Pseudo-Depth): Pikselin Y koordinatını derinlik olarak kullanır.
        Görüntünün alt kısmı (yüksek Y) "yakın", üst kısmı (düşük Y) "uzak" kabul edilir.
        Uzak bölgedeki gürültülü kenarları bastırır.
        """
        height, width = edges.shape
        
        # 1. Gradient Maskesi Oluştur (0.0 -> 1.0)
        # Üst %35'lik kısım tamamen "uzak" kabul edilip maskelenir (0)
        mask = np.zeros((height, width), dtype=np.float32)
        horizon = int(height * 0.35)
        
        # Numpy broadcasting ile hızlı gradient
        y_indices = np.arange(horizon, height)
        gradient_values = (y_indices - horizon) / (height - horizon)
        
        # Maskeyi doldur
        mask[horizon:, :] = gradient_values[:, np.newaxis]
        
        # 2. Kenarları Maske ile Ağırlıklandır
        weighted_edges = edges.astype(np.float32) * mask
        
        # 3. Threshold ile zayıflayan uzak kenarları temizle
        _, filtered_edges = cv2.threshold(weighted_edges, 50, 255, cv2.THRESH_BINARY)
        
        return filtered_edges.astype(np.uint8)

    def create_free_space_mask(self, bev_edges):
        """
        BEV kenar haritasından serbest alan maskesi oluşturur.
        """
        # Kenarları kalınlaştır (Dilation) - Engelleri belirginleştir
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(bev_edges, kernel, iterations=2)
        
        # Maskeyi ters çevir (Siyah: Engel, Beyaz: Boş Alan)
        free_space_mask = cv2.bitwise_not(dilated_edges)
        
        # Gürültü temizleme (Erosion) - Küçük beyaz noktaları kaldır
        free_space_mask = cv2.erode(free_space_mask, kernel, iterations=1)
        
        return free_space_mask

    def project_point(self, point):
        """
        Tek bir noktayı (x, y) IPM matrisi ile dönüştürür.
        """
        if self.ipm_matrix is None:
            return point
        
        # Noktayı homojen koordinatlara çevir: [x, y, 1]
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.ipm_matrix)
        
        return dst_point[0][0]

    def update_mask_with_obstacles(self, mask, obstacles):
        """
        YOLO engellerini maskeden çıkarır (Siyah yapar).
        """
        for item in obstacles:
            # item: (x1, y1, x2, y2, label, conf)
            x1, y1, x2, y2 = item[:4]
            
            # Engelin yere bastığı noktayı bul (Alt orta nokta)
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            
            # IPM ile BEV koordinatlarına çevir
            bev_point = self.project_point((bottom_center_x, bottom_center_y))
            bx, by = int(bev_point[0]), int(bev_point[1])
            
            # Maske sınırları içinde mi?
            if 0 <= bx < mask.shape[1] and 0 <= by < mask.shape[0]:
                # Maske üzerinde engeli işaretle (Siyah daire)
                # Engelin boyutuna göre yarıçap belirleyebiliriz ama şimdilik sabit güvenli alan
                cv2.circle(mask, (bx, by), 40, 0, -1) # 0 = Siyah (Engel)
            
        return mask

    def find_best_direction_weighted(self, edges, obstacles, width, height):
        """
        Ekranı 3 bölgeye (Sol, Orta, Sağ) ayırır.
        Her bölge için "Maliyet Skoru" hesaplar:
        Skor = (Kenar Yoğunluğu) + (Engel Cezası)
        En düşük skorlu (en güvenli) yönü seçer.
        """
        w_third = width // 3
        
        # 1. Kenar Yoğunluğu (Edge Density)
        # edges: 0-255 binary image. Beyaz piksel sayısı = Kenar miktarı
        left_score = np.sum(edges[:, :w_third]) / 255
        center_score = np.sum(edges[:, w_third:2*w_third]) / 255
        right_score = np.sum(edges[:, 2*w_third:]) / 255
        
        # 2. Nesne Cezası (Object Penalty)
        # Nesneler engelse skoru arttır
        obstacle_penalty = {"SOL": 0, "DÜZ": 0, "SAG": 0}
        
        for obs in obstacles:
            x1, y1, x2, y2, label, conf = obs
            cx = (x1 + x2) // 2
            
            # Nesnenin boyutu (Ekranı kaplama oranı) - Yakınlık göstergesi
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height
            ratio = box_area / frame_area
            
            # Ceza puanı: Yakınsa (büyükse) çok yüksek ceza
            # Örn: Ekranın %10'unu kaplayan bir engel 5000 puan ceza ekler
            penalty = ratio * 50000 
            
            if cx < w_third:
                obstacle_penalty["SOL"] += penalty
            elif cx < 2*w_third:
                obstacle_penalty["DÜZ"] += penalty
            else:
                obstacle_penalty["SAG"] += penalty
                
        # Toplam Skorlar
        scores = {
            "SOL": left_score + obstacle_penalty["SOL"],
            "DÜZ": center_score + obstacle_penalty["DÜZ"],
            "SAG": right_score + obstacle_penalty["SAG"]
        }
        
        # En düşük skorlu (en temiz) bölgeyi seç
        best_direction = min(scores, key=scores.get)
        
        # Eğer en iyi yolun bile skoru çok yüksekse DUR
        # Eşik değeri: Örn. bölgenin %30'u kenarlarla doluysa
        zone_area = (height * w_third)
        if scores[best_direction] > (zone_area * 0.3):
             return "DUR", scores
             
        return best_direction, scores

    def detect_obstacles(self, frame):
        """
        YOLO ile nesne tespiti yapar.
        """
        results = self.model(frame, verbose=False, conf=0.4)
        obstacles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    obstacles.append((x1, y1, x2, y2, label, conf))
        return obstacles

    def process_frame(self, frame, run_yolo=False):
        """
        Optimize Edilmiş Pipeline:
        YOLO -> Canny -> Pseudo-Depth -> Morphology -> Weighted Decision -> IPM -> Free-space
        """
        height, width = frame.shape[:2]

        # 1. YOLO (Önce çalışmalı ki karara etki etsin)
        if run_yolo:
            self.last_obstacles = self.detect_obstacles(frame)
        obstacles = self.last_obstacles

        # 2. Canny (Hafif Kenar Tespiti)
        edges = self.detect_edges(frame)
        
        # 3. Pseudo-Depth (Sahte 3D Filtreleme)
        depth_filtered_edges = self.apply_pseudo_depth(edges)
        
        # 4. Morphology (Kenar İyileştirme)
        morph_edges = self.apply_morphology(depth_filtered_edges)
        
        # 5. Karar (Weighted Region Logic)
        direction, scores = self.find_best_direction_weighted(morph_edges, obstacles, width, height)
        
        # 6. IPM (Kuş Bakışı - Görselleştirme için)
        bev_edges = self.apply_ipm(morph_edges)
        
        # 7. Free-space Mask (Görselleştirme için)
        free_space_mask = self.create_free_space_mask(bev_edges)
            
        # Görselleştirme
        combined_view = frame.copy()
        
        # Kenarları yeşil overlay yap
        combined_view[morph_edges > 0] = [0, 255, 0]
        
        # Yön bilgisini ve Skorları ekrana yaz
        cv2.putText(combined_view, f"ROT: {direction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Skorları göster
        cv2.putText(combined_view, f"L:{int(scores['SOL'])} C:{int(scores['DÜZ'])} R:{int(scores['SAG'])}", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # YOLO kutularını çiz
        for obs in obstacles:
            x1, y1, x2, y2, label, conf = obs
            cv2.rectangle(combined_view, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(combined_view, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        return combined_view, obstacles, bev_edges, free_space_mask, direction
