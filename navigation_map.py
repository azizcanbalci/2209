"""
3D Navigasyon Haritası ve Path Planning Modülü
Kör kullanıcılar için engelsiz yol bulma sistemi
"""

import cv2
import numpy as np
from collections import deque
import math

class NavigationMap:
    """
    2.5D Occupancy Grid Map + Path Planning
    Mono kameradan navigasyon haritası oluşturur
    """
    
    def __init__(self, grid_size=(100, 100), cell_size=0.1):
        """
        Args:
            grid_size: Harita boyutu (satır, sütun) - her hücre cell_size metre
            cell_size: Her hücrenin temsil ettiği alan (metre)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.cell_size = cell_size  # Her hücre 10cm
        
        # Occupancy Grid: 0=bilinmiyor, 1=boş, 2=engel
        self.occupancy_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        # Güvenilirlik haritası (0-255): Ne kadar emin olduğumuz
        self.confidence_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        # Kamera pozisyonu (grid koordinatlarında)
        # Kamera grid'in alt ortasında varsayılır
        self.camera_row = self.grid_rows - 5  # Alttan 5 hücre yukarıda
        self.camera_col = self.grid_cols // 2  # Ortada
        
        # Path için hedef noktası (varsayılan: düz ileri)
        self.target_row = 10  # Üst kısımda
        self.target_col = self.grid_cols // 2
        
        # Geçmiş haritalar (temporal smoothing için)
        self.grid_history = deque(maxlen=5)
        
        # Yürünebilir yol
        self.current_path = []
        
        # Görselleştirme renkleri
        self.colors = {
            'unknown': (50, 50, 50),      # Koyu gri
            'free': (0, 255, 0),          # Yeşil
            'obstacle': (0, 0, 255),      # Kırmızı
            'path': (255, 255, 0),        # Sarı
            'camera': (255, 0, 255),      # Mor
            'target': (0, 255, 255)       # Cyan
        }
    
    def reset_grid(self):
        """Haritayı sıfırla"""
        self.occupancy_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.confidence_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
    
    def frame_to_grid(self, x, y, frame_width, frame_height):
        """
        Frame koordinatlarını grid koordinatlarına çevirir.
        Perspektif dönüşümü ile yaklaşık 3D konum tahmini.
        
        Args:
            x, y: Frame koordinatları (piksel)
            frame_width, frame_height: Frame boyutları
        
        Returns:
            (grid_row, grid_col) veya None
        """
        # Normalize koordinatlar (0-1)
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Perspektif düzeltmesi
        # Üst kısım = uzak, alt kısım = yakın
        # y koordinatı derinliği temsil eder
        
        # Derinlik tahmini (y'ye göre)
        # y=0 (üst) -> uzak (row=0), y=1 (alt) -> yakın (row=grid_rows)
        depth_factor = norm_y  # 0=uzak, 1=yakın
        
        # Grid row (derinlik)
        # Perspektif: alt kısım daha geniş görünür
        grid_row = int((1 - depth_factor) * (self.grid_rows - 10))
        
        # X koordinatı perspektife göre düzeltilir
        # Alt kısımda X daha dar aralıkta, üstte daha geniş
        perspective_width = 0.3 + (1 - depth_factor) * 0.7  # 0.3-1.0 arası
        center_offset = (norm_x - 0.5) * perspective_width
        grid_col = int(self.grid_cols // 2 + center_offset * self.grid_cols)
        
        # Sınır kontrolü
        grid_row = max(0, min(grid_row, self.grid_rows - 1))
        grid_col = max(0, min(grid_col, self.grid_cols - 1))
        
        return grid_row, grid_col
    
    def update_from_obstacles(self, obstacles, frame_width, frame_height):
        """
        YOLO engellerinden haritayı güncelle.
        
        Args:
            obstacles: [(x1, y1, x2, y2), ...] formatında engel listesi
            frame_width, frame_height: Frame boyutları
        """
        # Önce tüm görünür alanı "boş" olarak işaretle (düşük güvenilirlik)
        visible_area = self.get_visible_area()
        for row, col in visible_area:
            if self.occupancy_grid[row, col] != 2:  # Engel değilse
                self.occupancy_grid[row, col] = 1  # Boş
                self.confidence_grid[row, col] = min(255, self.confidence_grid[row, col] + 10)
        
        # Engelleri haritaya ekle
        for (x1, y1, x2, y2) in obstacles:
            # Engelin alt orta noktası (zemine değdiği yer)
            cx = (x1 + x2) // 2
            cy = y2  # Alt kenar
            
            # Engel boyutuna göre grid'de kaplama alanı
            width = x2 - x1
            height = y2 - y1
            
            # Grid koordinatlarına çevir
            grid_pos = self.frame_to_grid(cx, cy, frame_width, frame_height)
            if grid_pos:
                row, col = grid_pos
                
                # Engel boyutuna göre genişlet
                size_factor = max(1, int((width / frame_width) * 10))
                
                # Engeli ve çevresini işaretle
                for dr in range(-size_factor, size_factor + 1):
                    for dc in range(-size_factor, size_factor + 1):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                            self.occupancy_grid[r, c] = 2  # Engel
                            self.confidence_grid[r, c] = min(255, self.confidence_grid[r, c] + 50)
    
    def update_from_free_space(self, free_space_mask, frame_width, frame_height):
        """
        Free space mask'tan boş alanları güncelle.
        
        Args:
            free_space_mask: Binary mask (255=boş, 0=dolu)
        """
        if free_space_mask is None:
            return
        
        # Mask'ı küçült (hız için)
        small_mask = cv2.resize(free_space_mask, (50, 50))
        
        # Her piksel için grid'e eşle
        h, w = small_mask.shape[:2]
        for y in range(h):
            for x in range(w):
                if small_mask[y, x] > 128:  # Boş alan
                    # Frame koordinatlarına ölçekle
                    fx = (x / w) * frame_width
                    fy = (y / h) * frame_height
                    
                    grid_pos = self.frame_to_grid(fx, fy, frame_width, frame_height)
                    if grid_pos:
                        row, col = grid_pos
                        if self.occupancy_grid[row, col] != 2:  # Engel değilse
                            self.occupancy_grid[row, col] = 1
                            self.confidence_grid[row, col] = min(255, self.confidence_grid[row, col] + 5)
    
    def get_visible_area(self):
        """Kameradan görünen alanın grid koordinatlarını döndürür."""
        visible = []
        # Üçgen görüş alanı (FOV yaklaşık 60 derece)
        for row in range(self.camera_row):
            # Görüş genişliği derinlikle artar
            depth = self.camera_row - row
            half_width = int(depth * 0.7)  # ~70 derece FOV
            
            for col in range(self.camera_col - half_width, self.camera_col + half_width + 1):
                if 0 <= col < self.grid_cols:
                    visible.append((row, col))
        return visible
    
    def find_safe_path(self):
        """
        Basit path planning: En yakın güvenli yolu bul.
        A* yerine daha basit greedy yaklaşım (real-time için).
        
        Returns:
            list: [(row, col), ...] yol noktaları
        """
        path = []
        current_row = self.camera_row
        current_col = self.camera_col
        
        # Hedefe doğru ilerle
        max_steps = 50
        for _ in range(max_steps):
            if current_row <= self.target_row:
                break
            
            # Bir sonraki adım için en iyi yönü bul
            best_col = current_col
            best_score = -float('inf')
            
            # Sol, düz, sağ kontrol et
            for dc in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                next_col = current_col + dc
                if 0 <= next_col < self.grid_cols:
                    # Skoru hesapla
                    score = 0
                    
                    # Engel kontrolü (önümüzdeki 3 hücre)
                    clear = True
                    for dr in range(1, 4):
                        check_row = current_row - dr
                        if 0 <= check_row < self.grid_rows:
                            if self.occupancy_grid[check_row, next_col] == 2:
                                clear = False
                                score -= 100  # Engel cezası
                                break
                            elif self.occupancy_grid[check_row, next_col] == 1:
                                score += 10  # Boş alan bonusu
                    
                    if clear:
                        # Hedefe yakınlık bonusu
                        dist_to_target = abs(next_col - self.target_col)
                        score -= dist_to_target * 2
                        
                        # Düz gitme bonusu
                        if dc == 0:
                            score += 5
                    
                    if score > best_score:
                        best_score = score
                        best_col = next_col
            
            # Hareket et
            current_row -= 1
            current_col = best_col
            path.append((current_row, current_col))
        
        self.current_path = path
        return path
    
    def get_navigation_command(self):
        """
        Path'e göre navigasyon komutu döndür.
        
        Returns:
            str: "DÜZ", "SOL", "SAG", "HAFIF_SOL", "HAFIF_SAG", "DUR"
        """
        if not self.current_path or len(self.current_path) < 3:
            # Düz ileri kontrol et
            clear_ahead = True
            for dr in range(1, 10):
                row = self.camera_row - dr
                if 0 <= row < self.grid_rows:
                    if self.occupancy_grid[row, self.camera_col] == 2:
                        clear_ahead = False
                        break
            
            if clear_ahead:
                return "DÜZ"
            else:
                return "DUR"
        
        # İlk birkaç adımın ortalamasına bak
        avg_col_diff = 0
        count = min(5, len(self.current_path))
        for i in range(count):
            avg_col_diff += self.current_path[i][1] - self.camera_col
        avg_col_diff /= count
        
        # Yön belirle
        if abs(avg_col_diff) < 2:
            return "DÜZ"
        elif avg_col_diff < -5:
            return "SOL"
        elif avg_col_diff > 5:
            return "SAG"
        elif avg_col_diff < 0:
            return "HAFIF_SOL"
        else:
            return "HAFIF_SAG"
    
    def get_obstacles_info(self):
        """
        Engel bilgilerini döndür (sesli anlatım için).
        
        Returns:
            dict: Engel konumları ve mesafeleri
        """
        info = {
            'left_obstacle': None,
            'center_obstacle': None,
            'right_obstacle': None
        }
        
        # Sol, orta, sağ bölgelerde en yakın engeli bul
        left_bound = self.grid_cols // 3
        right_bound = 2 * self.grid_cols // 3
        
        for col_range, key in [
            (range(0, left_bound), 'left_obstacle'),
            (range(left_bound, right_bound), 'center_obstacle'),
            (range(right_bound, self.grid_cols), 'right_obstacle')
        ]:
            min_dist = float('inf')
            for col in col_range:
                for row in range(self.camera_row - 1, -1, -1):
                    if self.occupancy_grid[row, col] == 2:
                        dist = (self.camera_row - row) * self.cell_size
                        if dist < min_dist:
                            min_dist = dist
                            info[key] = round(dist, 1)
                        break
        
        return info
    
    def visualize(self, scale=4):
        """
        Haritayı görselleştir.
        
        Args:
            scale: Büyütme faktörü
        
        Returns:
            np.ndarray: BGR görüntü
        """
        # Renkli harita oluştur
        vis = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        
        # Occupancy durumuna göre renklendir
        vis[self.occupancy_grid == 0] = self.colors['unknown']
        vis[self.occupancy_grid == 1] = self.colors['free']
        vis[self.occupancy_grid == 2] = self.colors['obstacle']
        
        # Path'i çiz
        for row, col in self.current_path:
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                vis[row, col] = self.colors['path']
        
        # Kamera pozisyonunu çiz
        cv2.circle(vis, (self.camera_col, self.camera_row), 2, self.colors['camera'], -1)
        
        # Hedef noktasını çiz
        cv2.circle(vis, (self.target_col, self.target_row), 2, self.colors['target'], -1)
        
        # Görüş alanı çizgileri
        # Sol ve sağ görüş sınırları
        fov_left = int(self.camera_col - (self.camera_row * 0.7))
        fov_right = int(self.camera_col + (self.camera_row * 0.7))
        cv2.line(vis, (self.camera_col, self.camera_row), (max(0, fov_left), 0), (100, 100, 100), 1)
        cv2.line(vis, (self.camera_col, self.camera_row), (min(self.grid_cols-1, fov_right), 0), (100, 100, 100), 1)
        
        # Büyüt
        vis = cv2.resize(vis, (self.grid_cols * scale, self.grid_rows * scale), interpolation=cv2.INTER_NEAREST)
        
        # Flip (üst=uzak, alt=yakın olsun)
        vis = cv2.flip(vis, 0)
        
        # Etiketler ekle
        cv2.putText(vis, "UZAK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "YAKIN", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "SOL", (10, vis.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "SAG", (vis.shape[1] - 40, vis.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def process_frame(self, obstacles, free_space_mask, frame_width, frame_height):
        """
        Bir frame'i işle ve haritayı güncelle.
        
        Args:
            obstacles: YOLO engelleri
            free_space_mask: Free space binary mask
            frame_width, frame_height: Frame boyutları
        
        Returns:
            tuple: (visualization, navigation_command, obstacles_info)
        """
        # Haritayı yumuşat (önceki değerleri koru)
        self.occupancy_grid = (self.occupancy_grid * 0.7).astype(np.uint8)
        
        # Engellerden güncelle
        self.update_from_obstacles(obstacles, frame_width, frame_height)
        
        # Free space'den güncelle
        self.update_from_free_space(free_space_mask, frame_width, frame_height)
        
        # Path bul
        self.find_safe_path()
        
        # Navigasyon komutu
        nav_command = self.get_navigation_command()
        
        # Engel bilgileri
        obstacles_info = self.get_obstacles_info()
        
        # Görselleştirme
        vis = self.visualize()
        
        return vis, nav_command, obstacles_info


class DepthEstimator:
    """
    Basit monoküler derinlik tahmini.
    MiDaS kullanmadan, nesne boyutuna dayalı yaklaşık derinlik.
    """
    
    def __init__(self):
        # Bilinen nesne boyutları (metre cinsinden ortalama yükseklik)
        self.known_heights = {
            'person': 1.7,
            'car': 1.5,
            'truck': 2.5,
            'bus': 3.0,
            'bicycle': 1.0,
            'motorcycle': 1.2,
            'dog': 0.5,
            'cat': 0.3,
            'chair': 0.9,
            'table': 0.75,
            'bottle': 0.25,
            'default': 1.0
        }
        
        # Kamera parametreleri (varsayılan değerler)
        self.focal_length = 600  # Piksel cinsinden odak uzaklığı (kalibrasyon gerekir)
    
    def estimate_depth(self, bbox_height, class_name, frame_height):
        """
        Nesne boyutuna göre derinlik tahmini.
        
        Formül: depth = (real_height * focal_length) / bbox_height
        
        Args:
            bbox_height: Bounding box yüksekliği (piksel)
            class_name: Nesne sınıfı
            frame_height: Frame yüksekliği
        
        Returns:
            float: Tahmini mesafe (metre)
        """
        if bbox_height <= 0:
            return 10.0
        
        # Gerçek yüksekliği al
        real_height = self.known_heights.get(class_name.lower(), self.known_heights['default'])
        
        # Derinlik hesapla
        depth = (real_height * self.focal_length) / bbox_height
        
        # Sınırla
        depth = max(0.5, min(depth, 20.0))
        
        return round(depth, 1)
    
    def create_depth_map(self, frame, obstacles):
        """
        Frame ve engellerden basit derinlik haritası oluştur.
        
        Args:
            frame: BGR görüntü
            obstacles: [(x1, y1, x2, y2, class_name, conf), ...]
        
        Returns:
            np.ndarray: Derinlik haritası (float32)
        """
        h, w = frame.shape[:2]
        depth_map = np.ones((h, w), dtype=np.float32) * 10.0  # Varsayılan 10m
        
        # Y koordinatına göre basit derinlik (perspektif)
        for y in range(h):
            # Üst = uzak, alt = yakın
            y_ratio = y / h
            base_depth = 15.0 * (1 - y_ratio) + 0.5 * y_ratio
            depth_map[y, :] = base_depth
        
        # Engelleri ekle
        for (x1, y1, x2, y2, class_name, conf) in obstacles:
            bbox_height = y2 - y1
            depth = self.estimate_depth(bbox_height, class_name, h)
            
            # Engel alanını güncelle
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            depth_map[y1:y2, x1:x2] = depth
        
        return depth_map
    
    def visualize_depth(self, depth_map):
        """
        Derinlik haritasını renkli görüntüye çevir.
        
        Args:
            depth_map: Float derinlik haritası
        
        Returns:
            np.ndarray: BGR görüntü
        """
        # Normalize (0-255)
        normalized = np.clip(depth_map / 15.0 * 255, 0, 255).astype(np.uint8)
        
        # Colormap uygula (yakın=kırmızı, uzak=mavi)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)
        
        return colored


if __name__ == "__main__":
    # Test
    nav_map = NavigationMap(grid_size=(100, 100))
    
    # Test engelleri
    test_obstacles = [
        (200, 300, 280, 400),  # Sol engel
        (350, 250, 450, 380),  # Orta engel
    ]
    
    # Güncelle
    nav_map.update_from_obstacles(test_obstacles, 640, 480)
    nav_map.find_safe_path()
    
    # Görselleştir
    vis = nav_map.visualize()
    cv2.imshow("Navigation Map", vis)
    
    print(f"Navigasyon komutu: {nav_map.get_navigation_command()}")
    print(f"Engel bilgileri: {nav_map.get_obstacles_info()}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
