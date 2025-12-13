import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter

class VisionPipeline:
    def __init__(self, model_path="../models/yolo11n.pt"):
        """
        Görüntü işleme pipeline'ını başlatır.
        """
        print(f"VisionPipeline başlatılıyor... Model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("YOLO modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            raise e
        
        # IPM Matrisleri
        self.ipm_matrix = None
        self.ipm_width = 0
        self.ipm_height = 0
        
        # --- AKIL KATMANI (Memory) ---
        # Son 7 karenin yön kararını saklar (Karar Stabilizasyonu için)
        self.direction_memory = deque(maxlen=7)
        
        # Zemin rengi adaptasyonu için hareketli ortalama
        self.floor_color_mean = None
        self.floor_color_std = None

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
        Canny kenar tespiti uygular.
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gürültü azaltmak için Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny Edge Detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

    def detect_smart_floor(self, frame):
        """
        AKILLI ZEMİN TESPİTİ (Adaptive Color + Edge Consensus)
        Sadece kenarlara bakmaz, zeminin rengini öğrenir ve ona göre karar verir.
        """
        h, w = frame.shape[:2]
        
        # 1. Ön İşleme
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 2. Referans Bölgesi (Ayak Ucu - Güvenli Alan)
        # Ekranın alt orta kısmı (100x100 piksel)
        roi_h, roi_w = 100, 100
        roi_y = h - roi_h
        roi_x = (w - roi_w) // 2
        sample_roi = hsv[roi_y:h, roi_x:roi_x+roi_w]
        
        # Anlık renk istatistikleri
        current_mean = np.mean(sample_roi, axis=(0, 1))
        current_std = np.std(sample_roi, axis=(0, 1))
        
        # Hafızalı Renk Öğrenme (Exponential Moving Average)
        if self.floor_color_mean is None:
            self.floor_color_mean = current_mean
            self.floor_color_std = current_std
        else:
            # Yeni rengi %10 oranında hafızaya kat (Ani değişimleri engelle)
            alpha = 0.1
            self.floor_color_mean = (1 - alpha) * self.floor_color_mean + alpha * current_mean
            self.floor_color_std = (1 - alpha) * self.floor_color_std + alpha * current_std
            
        # Dinamik Eşik Belirleme
        # Standart sapmanın 4 katı kadar esneklik tanı
        std_tolerance = np.maximum(self.floor_color_std, 10) # Min tolerans 10
        lower_bound = np.clip(self.floor_color_mean - (std_tolerance * 4), 0, 255)
        upper_bound = np.clip(self.floor_color_mean + (std_tolerance * 4), 0, 255)
        
        # 3. Renk Maskesi
        color_mask = cv2.inRange(hsv, lower_bound.astype(np.uint8), upper_bound.astype(np.uint8))
        
        # 4. Kenar Tespiti (Canny)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # 5. Konsensüs: Renk Uygun VE Kenar Değil
        floor_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(dilated_edges))
        
        # 6. Temizlik
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 7. En Büyük Parçayı Seç (Ana Zemin)
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Eğer çok küçükse (gürültü), boş döndür
            if cv2.contourArea(max_contour) > (h * w * 0.05):
                clean_mask = np.zeros_like(floor_mask)
                cv2.drawContours(clean_mask, [max_contour], -1, 255, -1)
                return clean_mask
                
        return floor_mask

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

    def find_best_direction(self, free_space_mask):
        """
        Free space maskesine göre en güvenli yönü belirler.
        AKILLI KARAR: Geçmiş kararları hatırlar ve titremeyi önler.
        """
        height, width = free_space_mask.shape
        
        # 3 şerit (Sol, Orta, Sağ)
        w_third = width // 3
        
        left_strip = free_space_mask[:, :w_third]
        center_strip = free_space_mask[:, w_third:2*w_third]
        right_strip = free_space_mask[:, 2*w_third:]
        
        # Her şeritteki beyaz piksel sayısı (Boş alan miktarı)
        scores = {
            "SOL": np.sum(left_strip == 255),
            "DÜZ": np.sum(center_strip == 255),
            "SAG": np.sum(right_strip == 255)
        }
        
        # Toplam alan (her şerit için yaklaşık)
        total_area = height * w_third
        
        # Doluluk oranları (Boş alan oranı)
        ratios = {k: v / total_area for k, v in scores.items()}
        
        current_decision = "DUR"
        
        # Eğer tüm yollar tıkalıysa (< %15 boş alan)
        if all(r < 0.15 for r in ratios.values()):
            current_decision = "DUR"
        else:
            # En iyi skoru bul
            best_direction = max(scores, key=scores.get)
            
            # "DÜZ" gitmek için hafif bir tolerans/öncelik tanıyalım
            if scores["DÜZ"] >= scores[best_direction] * 0.85:
                current_decision = "DÜZ"
            else:
                current_decision = best_direction
        
        # --- AKIL KATMANI (Karar Stabilizasyonu) ---
        self.direction_memory.append(current_decision)
        
        # Son 7 kararın en çok tekrar edeni (Mode)
        # Bu sayede anlık bir "DUR" veya "SOL" hatası sistemi yanıltmaz
        most_common_decision = Counter(self.direction_memory).most_common(1)[0][0]
        
        return most_common_decision

    def process_frame(self, frame):
        """
        Bir kareyi işler: YOLO tespiti + Akıllı Zemin Tespiti + IPM.
        """
        # 1. YOLO Nesne Tespiti (TRACKING MODU - Nesne Takibi)
        # persist=True: Nesne ID'lerini korur (Hafıza)
        results = self.model.track(frame, verbose=False, conf=0.4, persist=True)
        
        # 2. Akıllı Zemin Tespiti (Renk + Kenar)
        # Eski yöntem yerine bunu kullanıyoruz
        smart_floor_mask = self.detect_smart_floor(frame)
        
        # 3. Canny Kenar Tespiti (Görselleştirme ve yedek için)
        edges = self.detect_edges(frame)
        
        # 4. IPM Dönüşümü (Zemin Maskesi üzerinde)
        # Zemin maskesini kuş bakışına çevir
        bev_mask = self.apply_ipm(smart_floor_mask)
        bev_view = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        
        # Free Space Maskesi artık doğrudan BEV maskesi
        free_space_mask = bev_mask
        
        # Görselleştirme
        combined_view = frame.copy()
        
        # Zemini yeşil boya (Overlay)
        floor_overlay = np.zeros_like(frame)
        floor_overlay[smart_floor_mask == 255] = [0, 255, 0]
        combined_view = cv2.addWeighted(combined_view, 1.0, floor_overlay, 0.3, 0)
        
        obstacles = []
        
        # YOLO sonuçlarını işle
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Koordinatlar
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    
                    # ID varsa al (Tracking sayesinde)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    obstacles.append((x1, y1, x2, y2, label, conf))

        # 5. Hibrit Engel Haritası: YOLO engellerini maskeden çıkar
        free_space_mask = self.update_mask_with_obstacles(free_space_mask, obstacles)

        return combined_view, obstacles, edges, bev_view, free_space_mask
