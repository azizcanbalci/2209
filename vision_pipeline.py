"""
VisionPipeline - ONNX Runtime ile 32-bit ARM Uyumlu
Raspberry Pi 5 (32-bit Bookworm) için optimize edilmiş
"""
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque, Counter
import os

# COCO sınıf isimleri (YOLO modellerinde kullanılan)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class VisionPipeline:
    def __init__(self, model_path="../models/yolo11n.onnx"):
        """
        Görüntü işleme pipeline'ını başlatır.
        ONNX Runtime kullanarak 32-bit ARM uyumlu çalışır.
        
        Args:
            model_path: ONNX model dosyası yolu (.onnx uzantılı)
        """
        print(f"VisionPipeline başlatılıyor... Model: {model_path}")
        
        # ONNX dosyası kontrolü
        if not model_path.endswith('.onnx'):
            # .pt yerine .onnx uzantısı ekle
            model_path = model_path.replace('.pt', '.onnx')
            print(f"Model yolu düzeltildi: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model dosyası bulunamadı: {model_path}\n"
                "Lütfen önce modeli ONNX formatına çevirin:\n"
                "  from ultralytics import YOLO\n"
                "  model = YOLO('yolo11n.pt')\n"
                "  model.export(format='onnx', imgsz=640)"
            )
        
        try:
            # ONNX Runtime oturumu oluştur
            # CPU için optimize edilmiş provider
            providers = ['CPUExecutionProvider']
            
            # Session options (performans için)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # Raspberry Pi 5 için 4 thread
            
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            # Model input/output bilgileri
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            # Sınıf isimleri
            self.names = {i: name for i, name in enumerate(COCO_CLASSES)}
            
            print(f"[OK] ONNX model basariyla yuklendi!")
            print(f"  Input: {self.input_name} {self.input_shape}")
            print(f"  Outputs: {self.output_names}")
            
        except Exception as e:
            print(f"Model yukleme hatasi: {e}")
            raise e
        
        # Inference ayarları
        self.conf_threshold = 0.35  # Güven eşiği
        self.iou_threshold = 0.45   # NMS IoU eşiği
        self.input_size = 640       # Model giriş boyutu
        
        # IPM Matrisleri
        self.ipm_matrix = None
        self.ipm_width = 0
        self.ipm_height = 0
        
        # --- AKIL KATMANI (Memory) ---
        self.direction_memory = deque(maxlen=5)
        
        # Zemin rengi adaptasyonu
        self.floor_color_mean = None
        self.floor_color_std = None
        
        # Son tespit edilen engeller
        self.last_obstacles = []
        self.obstacle_history = deque(maxlen=3)

    def preprocess(self, frame):
        """
        YOLO ONNX modeli için görüntü ön işleme.
        """
        # Orijinal boyutları kaydet
        self.orig_h, self.orig_w = frame.shape[:2]
        
        # Letterbox resize (en-boy oranını koru)
        scale = min(self.input_size / self.orig_w, self.input_size / self.orig_h)
        new_w, new_h = int(self.orig_w * scale), int(self.orig_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding ekle (letterbox)
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Scaling faktörleri kaydet (postprocess için)
        self.scale = scale
        self.pad_w = pad_w
        self.pad_h = pad_h
        
        # BGR -> RGB, HWC -> CHW, normalize [0, 1]
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # Batch dimension
        
        return blob

    def postprocess(self, outputs, conf_threshold=None):
        """
        YOLO ONNX çıktısını işle (NMS dahil).
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        predictions = outputs[0]  # Shape: [1, 84, 8400] veya [1, 8400, 84]
        
        # Transpose if needed (bazı ONNX exportlar farklı format kullanır)
        if predictions.shape[1] == 84:
            predictions = predictions.transpose(0, 2, 1)  # [1, 8400, 84]
        
        predictions = predictions[0]  # [8400, 84]
        
        # boxes: [x_center, y_center, width, height]
        # scores: [class1_conf, class2_conf, ...]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Her kutu için en yüksek sınıf skoru
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Güven eşiği filtresi
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) == 0:
            return []
        
        # xywh -> xyxy dönüşümü
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Padding ve scale'i geri al (orijinal koordinatlara dönüş)
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - self.pad_w) / self.scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - self.pad_h) / self.scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - self.pad_w) / self.scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - self.pad_h) / self.scale
        
        # Sınırları kontrol et
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, self.orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, self.orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, self.orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, self.orig_h)
        
        # NMS uygula
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), 
            confidences.tolist(), 
            conf_threshold, 
            self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                conf = float(confidences[i])
                cls_id = int(class_ids[i])
                label = self.names.get(cls_id, f"class_{cls_id}")
                results.append((x1, y1, x2, y2, label, conf))
        
        return results

    def detect(self, frame):
        """
        Tek bir karede nesne tespiti yap.
        """
        # Ön işleme
        blob = self.preprocess(frame)
        
        # ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        # Son işleme
        detections = self.postprocess(outputs)
        
        return detections

    def init_ipm(self, width, height):
        """
        IPM (Inverse Perspective Mapping) matrisini hesaplar.
        """
        self.ipm_width = width
        self.ipm_height = height
        
        src_points = np.float32([
            [width * 0.25, height * 0.55],
            [width * 0.75, height * 0.55],
            [width * 0.05, height * 0.95],
            [width * 0.95, height * 0.95]
        ])
        
        dst_points = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

    def detect_smart_floor(self, frame):
        """
        AKILLI ZEMİN TESPİTİ (Adaptive Color + Edge Consensus)
        """
        h, w = frame.shape[:2]
        
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Referans Bölgesi (Ayak Ucu)
        roi_h, roi_w = 100, 100
        roi_y = h - roi_h
        roi_x = (w - roi_w) // 2
        sample_roi = hsv[roi_y:h, roi_x:roi_x+roi_w]
        
        current_mean = np.mean(sample_roi, axis=(0, 1))
        current_std = np.std(sample_roi, axis=(0, 1))
        
        if self.floor_color_mean is None:
            self.floor_color_mean = current_mean
            self.floor_color_std = current_std
        else:
            alpha = 0.1
            self.floor_color_mean = (1 - alpha) * self.floor_color_mean + alpha * current_mean
            self.floor_color_std = (1 - alpha) * self.floor_color_std + alpha * current_std
            
        std_tolerance = np.maximum(self.floor_color_std, 10)
        lower_bound = np.clip(self.floor_color_mean - (std_tolerance * 4), 0, 255)
        upper_bound = np.clip(self.floor_color_mean + (std_tolerance * 4), 0, 255)
        
        color_mask = cv2.inRange(hsv, lower_bound.astype(np.uint8), upper_bound.astype(np.uint8))
        
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        floor_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(dilated_edges))
        
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > (h * w * 0.05):
                clean_mask = np.zeros_like(floor_mask)
                cv2.drawContours(clean_mask, [max_contour], -1, 255, -1)
                return clean_mask
                
        return floor_mask

    def create_free_space_mask(self, bev_edges):
        """
        BEV kenar haritasından serbest alan maskesi oluşturur.
        """
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(bev_edges, kernel, iterations=2)
        free_space_mask = cv2.bitwise_not(dilated_edges)
        free_space_mask = cv2.erode(free_space_mask, kernel, iterations=1)
        return free_space_mask

    def project_point(self, point):
        """
        Tek bir noktayı IPM matrisi ile dönüştürür.
        """
        if self.ipm_matrix is None:
            return point
        
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.ipm_matrix)
        
        return dst_point[0][0]

    def update_mask_with_obstacles(self, mask, obstacles):
        """
        YOLO engellerini maskeden çıkarır.
        """
        for item in obstacles:
            x1, y1, x2, y2 = item[:4]
            
            w = x2 - x1
            shrink_factor = 0.3
            x1_shrunk = x1 + (w * shrink_factor / 2)
            x2_shrunk = x2 - (w * shrink_factor / 2)
            
            p1 = self.project_point((x1_shrunk, y2))
            p2 = self.project_point((x2_shrunk, y2))
            
            bx1, by1 = int(p1[0]), int(p1[1])
            bx2, by2 = int(p2[0]), int(p2[1])
            
            obstacle_width_bev = abs(bx2 - bx1)
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2 
            
            radius = max(15, obstacle_width_bev // 2)
            
            if 0 <= center_x < mask.shape[1] and 0 <= center_y < mask.shape[0]:
                cv2.circle(mask, (center_x, center_y), int(radius), 0, -1) 
            
        return mask

    def find_best_direction(self, free_space_mask, obstacles=None):
        """
        Free space maskesine göre en güvenli yönü belirler.
        """
        height, width = free_space_mask.shape
        
        weights = np.linspace(0.2, 1.5, height).reshape(-1, 1)
        weighted_mask = (free_space_mask / 255.0) * weights
        
        w_third = width // 3
        
        left_strip = weighted_mask[:, :w_third]
        center_strip = weighted_mask[:, w_third:2*w_third]
        right_strip = weighted_mask[:, 2*w_third:]
        
        scores = {
            "SOL": np.sum(left_strip),
            "DÜZ": np.sum(center_strip),
            "SAG": np.sum(right_strip)
        }
        
        bottom_half = free_space_mask[height//2:, :]
        bottom_left = np.sum(bottom_half[:, :w_third]) / 255.0
        bottom_center = np.sum(bottom_half[:, w_third:2*w_third]) / 255.0
        bottom_right = np.sum(bottom_half[:, 2*w_third:]) / 255.0
        
        bottom_scores = {
            "SOL": bottom_left,
            "DÜZ": bottom_center,
            "SAG": bottom_right
        }
        
        max_possible_score = np.sum(weights) * w_third
        max_bottom_score = (height // 2) * w_third
        
        ratios = {k: v / max_possible_score for k, v in scores.items()}
        bottom_ratios = {k: v / max_bottom_score for k, v in bottom_scores.items()}
        
        current_decision = "DUR"
        
        if all(r < 0.15 for r in bottom_ratios.values()):
            current_decision = "DUR"
        elif all(r < 0.15 for r in ratios.values()):
            current_decision = "DUR"
        else:
            hybrid_scores = {}
            for key in scores:
                hybrid_scores[key] = (scores[key] * 0.7) + (bottom_scores[key] * 0.3 * max_possible_score / max_bottom_score)
            
            best_direction = max(hybrid_scores, key=hybrid_scores.get)
            
            if hybrid_scores["DÜZ"] >= hybrid_scores[best_direction] * 0.75:
                current_decision = "DÜZ"
            else:
                current_decision = best_direction
        
        self.direction_memory.append(current_decision)
        
        if current_decision == "DUR":
            return "DUR"
        
        if len(self.direction_memory) >= 3:
            last_three = list(self.direction_memory)[-3:]
            if last_three.count(current_decision) >= 2:
                return current_decision
        
        most_common = Counter(self.direction_memory).most_common(1)[0][0]
        return most_common

    def process_frame(self, frame):
        """
        Bir kareyi işler: ONNX ile nesne tespiti + Akıllı Zemin Tespiti + IPM.
        """
        # 1. ONNX ile Nesne Tespiti
        obstacles = self.detect(frame)
        
        # 2. Akıllı Zemin Tespiti
        smart_floor_mask = self.detect_smart_floor(frame)
        
        # 3. Canny Kenar Tespiti
        edges = self.detect_edges(frame)
        
        # 4. IPM Dönüşümü
        bev_mask = self.apply_ipm(smart_floor_mask)
        bev_view = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        
        free_space_mask = bev_mask
        
        # Görselleştirme
        combined_view = frame.copy()
        
        # Zemini yeşil boya
        floor_overlay = np.zeros_like(frame)
        floor_overlay[smart_floor_mask == 255] = [0, 255, 0]
        combined_view = cv2.addWeighted(combined_view, 1.0, floor_overlay, 0.3, 0)
        
        # Engel geçmişini güncelle
        self.obstacle_history.append(len(obstacles))
        self.last_obstacles = obstacles

        # 5. YOLO engellerini maskeden çıkar
        free_space_mask = self.update_mask_with_obstacles(free_space_mask, obstacles)

        return combined_view, obstacles, edges, bev_view, free_space_mask
