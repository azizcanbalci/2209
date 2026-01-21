"""
RADAR NAVİGASYON SİSTEMİ
Kör kullanıcılar için gelişmiş engel radarı ve yol bulma
"""

import cv2
import numpy as np
from collections import deque
import math

class RadarNavigation:
    """
    Gelişmiş Radar Navigasyon Sistemi
    - Detaylı engel görselleştirmesi
    - Radar bazlı yön komutları
    - Kullanıcıyı (sarı) engellerden (kırmızı) uzak tutar
    """
    
    def __init__(self, radar_size=400):
        """
        Args:
            radar_size: Radar görüntüsü boyutu (piksel)
        """
        self.radar_size = radar_size
        self.radar_center = radar_size // 2
        self.max_distance = 8.0  # Maksimum mesafe (metre)
        
        # Kullanıcı pozisyonu (radarın alt ortasında)
        self.user_x = self.radar_center
        self.user_y = int(radar_size * 0.85)  # Alttan %15 yukarıda
        
        # Engel listesi: [(x, y, distance, size, region), ...]
        self.obstacles = []
        
        # Bölge skorları (engel yoğunluğu)
        self.region_scores = {
            'SOL': 0,
            'ORTA': 0,
            'SAG': 0
        }
        
        # En yakın engeller (bölgelere göre)
        self.closest_obstacles = {
            'SOL': None,
            'ORTA': None,
            'SAG': None
        }
        
        # Güvenli yol noktaları
        self.safe_path = []
        
        # Yön geçmişi (stabilizasyon için) - HIZLI VERSiYON
        self.direction_history = deque(maxlen=5)  # Azaltıldı: 10 -> 5
        self.stable_direction = "DÜZ"
        
        # Renk paleti
        self.colors = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'user': (0, 255, 255),        # Sarı - Kullanıcı
            'obstacle_close': (0, 0, 255),   # Kırmızı - Yakın engel
            'obstacle_medium': (0, 128, 255), # Turuncu - Orta engel
            'obstacle_far': (0, 255, 0),      # Yeşil - Uzak engel
            'safe_zone': (0, 100, 0),         # Koyu yeşil - Güvenli bölge
            'danger_zone': (0, 0, 100),       # Koyu kırmızı - Tehlikeli bölge
            'path': (255, 255, 0),            # Sarı - Güvenli yol
            'text': (255, 255, 255),
            'left_region': (255, 100, 100),   # Açık kırmızı
            'center_region': (100, 255, 100), # Açık yeşil
            'right_region': (100, 100, 255),  # Açık mavi
        }
    
    def clear(self):
        """Radar verilerini temizle"""
        self.obstacles = []
        self.region_scores = {'SOL': 0, 'ORTA': 0, 'SAG': 0}
        self.closest_obstacles = {'SOL': None, 'ORTA': None, 'SAG': None}
        self.safe_path = []
    
    def add_obstacle(self, x, y, distance, size, class_name="obstacle"):
        """
        Radara engel ekle.
        
        Args:
            x: Frame'deki X koordinatı (0-1 normalize)
            y: Frame'deki Y koordinatı (0-1 normalize)
            distance: Tahmini mesafe (metre)
            size: Engel boyutu (0-1 normalize)
            class_name: Engel sınıfı
        """
        # Bölge belirleme
        if x < 0.35:
            region = 'SOL'
        elif x > 0.65:
            region = 'SAG'
        else:
            region = 'ORTA'
        
        # Radar koordinatlarına çevir
        # X: sol=-1, orta=0, sağ=1 -> radar X
        radar_x = int(self.radar_center + (x - 0.5) * self.radar_size * 0.8)
        
        # Y: mesafeye göre (yakın=alt, uzak=üst)
        # distance 0-max_distance -> radar_y user_y'den yukarı
        dist_ratio = min(distance / self.max_distance, 1.0)
        radar_y = int(self.user_y - dist_ratio * (self.user_y - 30))
        
        self.obstacles.append({
            'radar_x': radar_x,
            'radar_y': radar_y,
            'distance': distance,
            'size': size,
            'region': region,
            'class_name': class_name,
            'frame_x': x,
            'frame_y': y
        })
        
        # Bölge skorunu güncelle (yakın engeller daha yüksek skor)
        danger_score = max(0, (3.0 - distance)) * (1 + size)  # Yakın + büyük = tehlikeli
        self.region_scores[region] += danger_score
        
        # En yakın engeli güncelle
        if self.closest_obstacles[region] is None or distance < self.closest_obstacles[region]['distance']:
            self.closest_obstacles[region] = self.obstacles[-1]
    
    def analyze_path_curvature(self):
        """
        Safe path'in kıvrımlarını analiz et.
        Her segment için yön değişimini hesapla.
        
        Returns:
            list: [(segment_index, turn_direction, turn_intensity), ...]
        """
        if len(self.safe_path) < 3:
            return []
        
        curves = []
        for i in range(1, len(self.safe_path) - 1):
            prev_pt = self.safe_path[i - 1]
            curr_pt = self.safe_path[i]
            next_pt = self.safe_path[i + 1]
            
            # Önceki ve sonraki segment vektörleri
            vec1_x = curr_pt[0] - prev_pt[0]
            vec1_y = curr_pt[1] - prev_pt[1]
            vec2_x = next_pt[0] - curr_pt[0]
            vec2_y = next_pt[1] - curr_pt[1]
            
            # X değişimi (kıvrım yönü)
            x_change = vec2_x - vec1_x
            
            if abs(x_change) > 5:  # Anlamlı kıvrım
                direction = "SOL" if x_change < 0 else "SAG"
                intensity = min(abs(x_change) / 20, 1.0)  # 0-1 arası yoğunluk
                curves.append((i, direction, intensity))
        
        return curves
    
    def get_upcoming_turn(self):
        """
        Yaklaşan kıvrımı tespit et.
        
        Returns:
            tuple: (turn_direction, distance_to_turn, intensity) veya None
        """
        curves = self.analyze_path_curvature()
        
        if not curves:
            return None
        
        # İlk anlamlı kıvrımı bul (yoğunluğu 0.3'ten büyük)
        for segment_idx, direction, intensity in curves:
            if intensity > 0.3:
                # Kıvrıma olan mesafe (segment sayısı * adım uzunluğu)
                distance = segment_idx * 0.5  # Her segment ~0.5m
                return (direction, distance, intensity)
        
        return None
    
    def calculate_safe_direction(self):
        """
        SAFE PATH KIVIRIMLARINI TAKİP EDEN YÖN HESAPLAMA - KESİN VERSİYON
        Radar'daki sarı yolu kıvrımlarıyla birlikte takip eder.
        Daha düşük eşiklerle daha hızlı karar verir.
        
        Returns:
            str: "DÜZ", "SOL", "SAG", "HAFIF_SOL", "HAFIF_SAG", "DUR"
        """
        # Önce safe path'i hesapla
        self.calculate_safe_path()
        
        # En yakın engel mesafeleri
        sol_dist = self.closest_obstacles['SOL']['distance'] if self.closest_obstacles['SOL'] else 10
        orta_dist = self.closest_obstacles['ORTA']['distance'] if self.closest_obstacles['ORTA'] else 10
        sag_dist = self.closest_obstacles['SAG']['distance'] if self.closest_obstacles['SAG'] else 10
        
        # ACİL DURUM: Ortada çok yakın engel (1.2m içinde) - biraz artırıldı
        if orta_dist < 1.2:
            if sol_dist > sag_dist and sol_dist > 1.2:
                return "SOL"
            elif sag_dist > 1.2:
                return "SAG"
            else:
                return "DUR"
        
        # ORTA MESAFE ENGELİ: 1.2-2.5m arası - yönlendirme başlasın
        if orta_dist < 2.5:
            if sol_dist > sag_dist and sol_dist > orta_dist * 1.3:
                return "HAFIF_SOL"
            elif sag_dist > orta_dist * 1.3:
                return "HAFIF_SAG"
        
        # SAFE PATH KIVIRIMLARINI ANALİZ ET
        if len(self.safe_path) >= 3:
            # Yaklaşan kıvrımı kontrol et
            upcoming_turn = self.get_upcoming_turn()
            
            if upcoming_turn:
                turn_dir, turn_dist, turn_intensity = upcoming_turn
                
                # Kıvrım çok yakınsa (1.5m içinde) - hemen dön
                if turn_dist < 1.5:
                    if turn_intensity > 0.5:  # Düşürüldü: 0.6 -> 0.5
                        return turn_dir
                    else:
                        return f"HAFIF_{turn_dir}"
                
                # Kıvrım biraz ileride (1.5-3m) - hafif dönmeye başla
                elif turn_dist < 3.0 and turn_intensity > 0.4:  # Düşürüldü: 0.5 -> 0.4
                    return f"HAFIF_{turn_dir}"
            
            # Kıvrım yoksa - mevcut yol yönünü takip et
            path_points = self.safe_path[:min(4, len(self.safe_path))]  # Azaltıldı: 5 -> 4
            
            # Ağırlıklı X kayması hesapla
            total_x_shift = 0
            total_weight = 0
            for i, (px, py) in enumerate(path_points):
                x_diff = px - self.user_x
                weight = 1.0 + (len(path_points) - i) * 0.7  # Artırıldı: 0.5 -> 0.7
                total_x_shift += x_diff * weight
                total_weight += weight
            
            avg_x_shift = total_x_shift / total_weight if total_weight > 0 else 0
            
            # X kaymasına göre yön belirle - eşikler düşürüldü
            if avg_x_shift < -20:  # Düşürüldü: -25 -> -20
                return "SOL"
            elif avg_x_shift < -6:  # Düşürüldü: -8 -> -6
                return "HAFIF_SOL"
            elif avg_x_shift > 20:  # Düşürüldü: 25 -> 20
                return "SAG"
            elif avg_x_shift > 6:  # Düşürüldü: 8 -> 6
                return "HAFIF_SAG"
            else:
                return "DÜZ"
        
        # Safe path yoksa bölge skorlarına bak
        sol_score = self.region_scores['SOL']
        orta_score = self.region_scores['ORTA']
        sag_score = self.region_scores['SAG']
        
        # Hiç engel yoksa düz git
        if sol_score == 0 and orta_score == 0 and sag_score == 0:
            return "DÜZ"
        
        # Ortada engel yoksa düz
        if orta_score < 1.0 and orta_dist > 3.0:
            return "DÜZ"
        
        # En güvenli bölgeye yönlen
        if sol_score < orta_score and sol_score < sag_score:
            return "SOL" if sol_score < orta_score * 0.5 else "HAFIF_SOL"
        elif sag_score < orta_score:
            return "SAG" if sag_score < orta_score * 0.5 else "HAFIF_SAG"
        
        return "DÜZ"
    
    def get_stable_direction(self):
        """
        Stabilize edilmiş yön komutu.
        HIZLI VERSİYON - Daha kısa geçmiş, daha hızlı tepki.
        """
        raw_direction = self.calculate_safe_direction()
        self.direction_history.append(raw_direction)
        
        # ACİL DURUMLAR HEMEN GEÇMELİ
        if raw_direction == "DUR":
            self.stable_direction = "DUR"
            return "DUR"
        
        # Son 5 yönün çoğunluğunu bul (Azaltıldı: 10 -> 5)
        if len(self.direction_history) >= 3:
            from collections import Counter
            recent = list(self.direction_history)[-5:]  # Son 5
            counts = Counter(recent)
            most_common = counts.most_common(1)[0]
            
            # %30 çoğunluk yeterli (Azaltıldı: %40 -> %30)
            if most_common[1] >= len(recent) * 0.30:
                self.stable_direction = most_common[0]
        
        return self.stable_direction
    
    def calculate_safe_path(self):
        """
        Engelsiz güvenli yolu hesapla.
        """
        self.safe_path = []
        
        # Başlangıç noktası (kullanıcı)
        current_x = self.user_x
        current_y = self.user_y
        
        # Hedefe doğru ilerle (yukarı)
        target_y = 50  # Üst kısım
        step_size = 10
        
        while current_y > target_y:
            # En güvenli X pozisyonunu bul
            best_x = current_x
            min_danger = float('inf')
            
            for test_x in range(50, self.radar_size - 50, 10):
                # Bu noktadaki tehlike skorunu hesapla
                danger = 0
                for obs in self.obstacles:
                    dist = math.sqrt((test_x - obs['radar_x'])**2 + (current_y - step_size - obs['radar_y'])**2)
                    if dist < 50:
                        danger += (50 - dist) / 50 * (3.0 / max(0.5, obs['distance']))
                
                # Merkeze yakınlık bonusu
                center_penalty = abs(test_x - self.radar_center) * 0.01
                danger += center_penalty
                
                if danger < min_danger:
                    min_danger = danger
                    best_x = test_x
            
            current_y -= step_size
            current_x = int(current_x * 0.7 + best_x * 0.3)  # Yumuşak geçiş
            self.safe_path.append((current_x, current_y))
    
    def draw_radar(self):
        """
        Detaylı radar görüntüsü oluştur.
        
        Returns:
            np.ndarray: BGR radar görüntüsü
        """
        # Arka plan
        radar = np.zeros((self.radar_size, self.radar_size, 3), dtype=np.uint8)
        radar[:] = self.colors['background']
        
        # Grid çizgileri (mesafe halkaları)
        for dist in [2, 4, 6, 8]:  # metre
            radius = int((dist / self.max_distance) * (self.user_y - 30))
            cv2.circle(radar, (self.user_x, self.user_y), radius, self.colors['grid'], 1)
            # Mesafe etiketi
            cv2.putText(radar, f"{dist}m", (self.user_x + radius - 15, self.user_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['grid'], 1)
        
        # Bölge çizgileri
        left_line = int(self.radar_size * 0.35)
        right_line = int(self.radar_size * 0.65)
        cv2.line(radar, (left_line, 0), (left_line, self.user_y), self.colors['grid'], 1)
        cv2.line(radar, (right_line, 0), (right_line, self.user_y), self.colors['grid'], 1)
        
        # Bölge etiketleri
        cv2.putText(radar, "SOL", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(radar, "ORTA", (self.radar_center - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(radar, "SAG", (self.radar_size - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Bölge tehlike göstergesi (arka plan rengi)
        for region, x_range in [('SOL', (0, left_line)), ('ORTA', (left_line, right_line)), ('SAG', (right_line, self.radar_size))]:
            score = self.region_scores[region]
            if score > 5:  # Tehlikeli
                color = (0, 0, min(100, int(score * 20)))  # Kırmızı ton
            elif score > 2:  # Orta
                color = (0, min(100, int(score * 30)), min(100, int(score * 20)))  # Turuncu ton
            else:  # Güvenli
                color = (0, min(50, int((5 - score) * 10)), 0)  # Yeşil ton
            
            overlay = radar.copy()
            cv2.rectangle(overlay, (x_range[0], 0), (x_range[1], self.user_y - 20), color, -1)
            radar = cv2.addWeighted(radar, 0.7, overlay, 0.3, 0)
        
        # Güvenli yolu çiz (KALIN SARI ÇİZGİ)
        self.calculate_safe_path()
        if len(self.safe_path) > 1:
            # Yol çizgisi - segment segment farklı kalınlıkta
            for i in range(len(self.safe_path) - 1):
                pt1 = self.safe_path[i]
                pt2 = self.safe_path[i + 1]
                # Yakın segmentler daha kalın
                thickness = max(2, 4 - i // 5)
                cv2.line(radar, pt1, pt2, self.colors['path'], thickness)
            
            # Yol başlangıcında ok işareti (yönü göster)
            if len(self.safe_path) >= 2:
                start_pt = (self.user_x, self.user_y - 20)
                end_pt = self.safe_path[min(3, len(self.safe_path)-1)]
                cv2.arrowedLine(radar, start_pt, end_pt, (0, 255, 255), 3, tipLength=0.3)
            
            # KIVIRIM NOKTALARINI İŞARETLE
            curves = self.analyze_path_curvature()
            for segment_idx, turn_dir, intensity in curves:
                if intensity > 0.3 and segment_idx < len(self.safe_path):
                    curve_pt = self.safe_path[segment_idx]
                    # Kıvrım noktasını işaretle (turuncu daire)
                    curve_color = (0, 165, 255)  # Turuncu
                    radius = int(5 + intensity * 10)
                    cv2.circle(radar, curve_pt, radius, curve_color, 2)
                    
                    # Kıvrım yönünü göster
                    arrow_offset = 15 if turn_dir == "SAG" else -15
                    cv2.arrowedLine(radar, curve_pt, 
                                   (curve_pt[0] + arrow_offset, curve_pt[1]),
                                   curve_color, 2, tipLength=0.5)
        
        # Yaklaşan kıvrım uyarısı
        upcoming = self.get_upcoming_turn()
        if upcoming:
            turn_dir, turn_dist, turn_int = upcoming
            if turn_dist < 3.0:  # 3m içindeki kıvrımlar
                warn_text = f"KIVIRIM: {turn_dir} ({turn_dist:.1f}m)"
                warn_color = (0, 165, 255) if turn_dist > 1.5 else (0, 0, 255)
                cv2.putText(radar, warn_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, warn_color, 1)
        
        # Engelleri çiz
        for obs in self.obstacles:
            x, y = obs['radar_x'], obs['radar_y']
            dist = obs['distance']
            size = max(5, int(obs['size'] * 30))
            
            # Mesafeye göre renk
            if dist < 2.0:
                color = self.colors['obstacle_close']  # Kırmızı
                thickness = -1  # Dolu
            elif dist < 4.0:
                color = self.colors['obstacle_medium']  # Turuncu
                thickness = -1
            else:
                color = self.colors['obstacle_far']  # Yeşil
                thickness = 2  # Çerçeve
            
            # Engel çiz
            cv2.circle(radar, (x, y), size, color, thickness)
            
            # Mesafe etiketi (yakın engeller için)
            if dist < 4.0:
                cv2.putText(radar, f"{dist:.1f}m", (x - 15, y - size - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Kullanıcı (BİZ) - Sarı üçgen
        user_pts = np.array([
            [self.user_x, self.user_y - 15],      # Üst nokta
            [self.user_x - 12, self.user_y + 10], # Sol alt
            [self.user_x + 12, self.user_y + 10]  # Sağ alt
        ], np.int32)
        cv2.fillPoly(radar, [user_pts], self.colors['user'])
        cv2.putText(radar, "SEN", (self.user_x - 15, self.user_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['user'], 1)
        
        # Görüş alanı çizgileri
        fov_angle = 60  # derece
        fov_length = self.user_y - 30
        left_angle = math.radians(90 + fov_angle / 2)
        right_angle = math.radians(90 - fov_angle / 2)
        
        left_end = (int(self.user_x + fov_length * math.cos(left_angle)),
                   int(self.user_y - fov_length * math.sin(left_angle)))
        right_end = (int(self.user_x + fov_length * math.cos(right_angle)),
                    int(self.user_y - fov_length * math.sin(right_angle)))
        
        cv2.line(radar, (self.user_x, self.user_y), left_end, (100, 100, 100), 1)
        cv2.line(radar, (self.user_x, self.user_y), right_end, (100, 100, 100), 1)
        
        # Yön komutu
        direction = self.get_stable_direction()
        dir_colors = {
            'DÜZ': (0, 255, 0),
            'SOL': (255, 100, 100),
            'SAG': (100, 100, 255),
            'HAFIF_SOL': (200, 150, 100),
            'HAFIF_SAG': (100, 150, 200),
            'DUR': (0, 0, 255)
        }
        dir_color = dir_colors.get(direction, (255, 255, 255))
        
        # Yön ok işareti
        arrow_y = self.radar_size - 50
        if direction == "DÜZ":
            cv2.arrowedLine(radar, (self.radar_center, arrow_y + 20), 
                          (self.radar_center, arrow_y - 20), dir_color, 3, tipLength=0.5)
        elif direction in ["SOL", "HAFIF_SOL"]:
            offset = -40 if direction == "SOL" else -20
            cv2.arrowedLine(radar, (self.radar_center, arrow_y), 
                          (self.radar_center + offset, arrow_y - 15), dir_color, 3, tipLength=0.4)
        elif direction in ["SAG", "HAFIF_SAG"]:
            offset = 40 if direction == "SAG" else 20
            cv2.arrowedLine(radar, (self.radar_center, arrow_y), 
                          (self.radar_center + offset, arrow_y - 15), dir_color, 3, tipLength=0.4)
        elif direction == "DUR":
            cv2.rectangle(radar, (self.radar_center - 20, arrow_y - 15),
                         (self.radar_center + 20, arrow_y + 15), dir_color, -1)
        
        # Komut yazısı
        cv2.putText(radar, direction, (self.radar_center - 30, self.radar_size - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)
        
        # Bölge skorları (debug)
        score_y = self.radar_size - 80
        cv2.putText(radar, f"Sol:{self.region_scores['SOL']:.1f}", (10, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        cv2.putText(radar, f"Orta:{self.region_scores['ORTA']:.1f}", (self.radar_center - 30, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        cv2.putText(radar, f"Sag:{self.region_scores['SAG']:.1f}", (self.radar_size - 70, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        
        return radar
    
    def process_frame(self, obstacles, frame_width, frame_height):
        """
        Frame'den engelleri işle ve radar güncelle.
        
        Args:
            obstacles: [(x1, y1, x2, y2, class_name, distance), ...] veya [(x1, y1, x2, y2), ...]
            frame_width, frame_height: Frame boyutları
        
        Returns:
            tuple: (radar_image, direction_command, obstacles_info)
        """
        self.clear()
        
        for obs in obstacles:
            if len(obs) >= 6:
                x1, y1, x2, y2, class_name, distance = obs[:6]
            elif len(obs) >= 4:
                x1, y1, x2, y2 = obs[:4]
                class_name = "obstacle"
                # Boyuta göre mesafe tahmini
                bbox_height = y2 - y1
                y_ratio = y2 / frame_height
                size_ratio = bbox_height / frame_height
                
                # Basit mesafe tahmini
                if size_ratio < 0.1:
                    distance = 6.0
                elif size_ratio < 0.2:
                    distance = 4.0
                elif size_ratio < 0.35:
                    distance = 2.5
                else:
                    distance = 1.0 + (1 - y_ratio) * 2
            else:
                continue
            
            # Normalize koordinatlar
            cx = (x1 + x2) / 2 / frame_width
            cy = y2 / frame_height  # Alt kenar
            size = (x2 - x1) / frame_width
            
            self.add_obstacle(cx, cy, distance, size, class_name)
        
        # Radar çiz
        radar_img = self.draw_radar()
        
        # Yön komutu
        direction = self.get_stable_direction()
        
        # Engel bilgileri
        obstacles_info = {
            'sol': self.closest_obstacles['SOL']['distance'] if self.closest_obstacles['SOL'] else None,
            'orta': self.closest_obstacles['ORTA']['distance'] if self.closest_obstacles['ORTA'] else None,
            'sag': self.closest_obstacles['SAG']['distance'] if self.closest_obstacles['SAG'] else None,
            'scores': self.region_scores.copy()
        }
        
        return radar_img, direction, obstacles_info


# Test
if __name__ == "__main__":
    radar = RadarNavigation(radar_size=400)
    
    # Test engelleri
    test_obstacles = [
        (100, 200, 180, 350, "person", 2.5),  # Sol yakın
        (300, 150, 400, 300, "car", 4.0),     # Orta uzak
        (500, 250, 580, 400, "chair", 1.5),   # Sağ çok yakın
    ]
    
    radar_img, direction, info = radar.process_frame(test_obstacles, 640, 480)
    
    print(f"Yön: {direction}")
    print(f"Engel bilgileri: {info}")
    
    cv2.imshow("Radar", radar_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
