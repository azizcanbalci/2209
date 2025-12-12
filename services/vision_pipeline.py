import cv2
import numpy as np
from ultralytics import YOLO
import torch
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
            
        # 2. MiDaS Depth Modelini Yükle (Torch Hub)
        print("MiDaS Depth modeli yükleniyor...")
        try:
            self.midas_type = "MiDaS_small"  # Hız için small model
            self.midas = torch.hub.load("intel-isl/MiDaS", self.midas_type)
            
            # GPU varsa kullan
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.midas.to(self.device)
            self.midas.eval()
            
            # Transform işlemleri
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform if self.midas_type == "MiDaS_small" else midas_transforms.dpt_transform
            print(f"MiDaS ({self.device}) başarıyla yüklendi.")
        except Exception as e:
            print(f"MiDaS yükleme hatası: {e}")
            self.midas = None

        # 3. HED (Holistically-Nested Edge Detection) Modelini Yükle
        # Not: HED için 'deploy.prototxt' ve 'hed_pretrained_bsds.caffemodel' dosyaları gerekir.
        # Eğer yoksa Canny fallback olarak kullanılacak.
        self.hed_net = None
        try:
            hed_proto = "models/deploy.prototxt"
            hed_model = "models/hed_pretrained_bsds.caffemodel"
            if os.path.exists(hed_proto) and os.path.exists(hed_model):
                self.hed_net = cv2.dnn.readNetFromCaffe(hed_proto, hed_model)
                print("HED modeli başarıyla yüklendi.")
            else:
                print("HED model dosyaları bulunamadı, Canny kullanılacak.")
        except Exception as e:
            print(f"HED yükleme hatası: {e}")

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

    def estimate_depth(self, frame):
        """
        MiDaS ile derinlik haritası oluşturur.
        """
        if self.midas is None:
            return None
            
        # Görüntüyü hazırla
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        # Tahmin
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Orijinal boyuta resize et
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Normalize et (0-255)
        depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_map_norm

    def detect_edges(self, frame, low_threshold=50, high_threshold=150):
        """
        Kenar tespiti uygular (HED varsa HED, yoksa Canny).
        """
        if self.hed_net is not None:
            # HED Inference
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(500, 500),
                                       mean=(104.00698793, 116.66876762, 122.67891434),
                                       swapRB=False, crop=False)
            self.hed_net.setInput(blob)
            hed_output = self.hed_net.forward()
            
            # Çıktıyı işle
            hed_output = hed_output[0, 0]
            hed_output = cv2.resize(hed_output, (frame.shape[1], frame.shape[0]))
            hed_output = (255 * hed_output).astype("uint8")
            return hed_output
        else:
            # Fallback: Canny
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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

    def create_fusion_map(self, depth_map, edges, obstacles):
        """
        YOLO + HED + Depth Fusion: 3D Edge-Aware Detection
        """
        if depth_map is None:
            return edges, None # Depth yoksa sadece kenarları döndür
            
        # 1. Depth Thresholding: Sadece "yakın" nesneleri al
        # MiDaS'ta parlak = yakın, karanlık = uzak
        # Görüntünün en parlak %30'luk kısmını "yakın" kabul edelim
        threshold = np.percentile(depth_map, 70)
        near_mask = cv2.threshold(depth_map, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        
        # 2. Edge Filtering: Sadece yakındaki kenarları tut
        # Uzaktaki karmaşık arka plan kenarlarını eler
        fusion_edges = cv2.bitwise_and(edges, edges, mask=near_mask)
        
        # 3. YOLO Obstacle Integration
        # YOLO kutularını "kesin engel" olarak ekle
        obstacle_mask = np.zeros_like(edges)
        for obs in obstacles:
            x1, y1, x2, y2 = obs[:4]
            cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)
            
        # Fusion: Yakın Kenarlar + YOLO Engelleri
        fusion_map = cv2.bitwise_or(fusion_edges, obstacle_mask)
        
        return fusion_map, near_mask

    def process_frame(self, frame):
        """
        Bir kareyi işler: YOLO + HED + Depth Fusion.
        """
        # 1. YOLO Nesne Tespiti
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

        # 2. Edge Detection (HED or Canny)
        edges = self.detect_edges(frame)
        
        # 3. Depth Estimation (MiDaS)
        depth_map = self.estimate_depth(frame)
        
        # 4. Fusion (YOLO + Edges + Depth)
        fusion_map, near_mask = self.create_fusion_map(depth_map, edges, obstacles)
        
        # 5. IPM ve Free Space (Fusion Map üzerinden)
        # Artık ham kenarlar yerine "Fusion Map"i kuş bakışına çeviriyoruz
        bev_fusion = self.apply_ipm(fusion_map)
        
        # Free Space Maskesi (Fusion Map üzerinden daha temiz çıkar)
        free_space_mask = self.create_free_space_mask(bev_fusion)
        
        # Görselleştirme
        combined_view = frame.copy()
        
        # Depth haritasını renklendir (Görselleştirme için)
        depth_colormap = None
        if depth_map is not None:
            depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
        # Fusion Map'i yeşil overlay yap
        combined_view[fusion_map > 0] = [0, 255, 0]
        
        # YOLO kutularını çiz
        for obs in obstacles:
            x1, y1, x2, y2, label, conf = obs
            cv2.rectangle(combined_view, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(combined_view, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        return combined_view, obstacles, fusion_map, bev_fusion, free_space_mask, depth_colormap
