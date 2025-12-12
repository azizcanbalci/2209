import cv2
import numpy as np
from ultralytics import YOLO

class VisionPipeline:
    def __init__(self, model_path="../models/yolo11n.pt"):
        """
        Görüntü işleme pipeline'ını başlatır.
        
        Args:
            model_path: YOLO model dosyasının yolu
        """
        print(f"VisionPipeline başlatılıyor... Model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("YOLO modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            raise e
        
        # IPM Matrisleri (Lazy initialization)
        self.ipm_matrix = None
        self.ipm_width = 0
        self.ipm_height = 0

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
        
        # Eğer tüm yollar tıkalıysa (< %10 boş alan)
        if all(r < 0.1 for r in ratios.values()):
            return "DUR"
            
        # En iyi skoru bul
        best_direction = max(scores, key=scores.get)
        
        # "DÜZ" gitmek için hafif bir tolerans/öncelik tanıyalım
        # Eğer DÜZ skoru, en iyinin %85'inden fazlaysa DÜZ gitmeyi tercih et
        # Bu, sürekli küçük farklarla sağa/sola zikzak yapmayı engeller
        if scores["DÜZ"] >= scores[best_direction] * 0.85:
            return "DÜZ"
            
        return best_direction

    def process_frame(self, frame):
        """
        Bir kareyi işler: YOLO tespiti + Canny kenar tespiti + IPM + Free Space.
        
        Returns:
            combined_view: İşlenmiş görüntü (görselleştirme için)
            obstacles: Tespit edilen engellerin listesi
            edges: Kenar haritası
            bev_view: Kuş bakışı görünüm (IPM uygulanmış kenarlar)
            free_space_mask: Serbest alan maskesi
        """
        # 1. YOLO Nesne Tespiti
        results = self.model(frame, verbose=False, conf=0.4)
        
        # 2. Canny Kenar Tespiti
        edges = self.detect_edges(frame)
        
        # 3. IPM Dönüşümü (Kenar haritası üzerinde)
        bev_edges = self.apply_ipm(edges)
        bev_view = cv2.cvtColor(bev_edges, cv2.COLOR_GRAY2BGR)
        
        # 4. Free Space Maskesi Oluşturma
        free_space_mask = self.create_free_space_mask(bev_edges)
        
        # Görselleştirme için kenar haritasını BGR'ye çevir
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Orijinal görüntü ile kenarları birleştir (Overlay)
        # Kenar olan yerleri vurgula
        combined_view = frame.copy()
        combined_view[edges > 0] = [0, 255, 0]  # Kenarları yeşil yap
        
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
                    
                    obstacles.append((x1, y1, x2, y2, label, conf))
                    
                    # Not: Çizim işlemleri ana döngüde yapılacak (mesafe bilgisi için)

        # 5. Hibrit Engel Haritası: YOLO engellerini maskeden çıkar
        free_space_mask = self.update_mask_with_obstacles(free_space_mask, obstacles)

        return combined_view, obstacles, edges, bev_view, free_space_mask
