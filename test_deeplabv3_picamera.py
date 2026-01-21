"""
DeepLabV3 Semantic Segmentation Test - Pi Camera ile

Bu script, Raspberry Pi Camera Module v3 ile DeepLabV3 Cityscapes
modelini kullanarak gerçek zamanlı semantic segmentation yapar.

Model: models/deeplab_cityscapes.tflite
Girdi: 513x513 RGB görüntü
Çıktı: Her piksel için sınıf tahmini (19 Cityscapes sınıfı)

Kullanım:
    python test_deeplabv3_picamera.py

Tuşlar:
    'q' - Çıkış
    's' - Ekran görüntüsü kaydet
    'p' - Segmentasyonu duraklat/devam ettir
"""

import cv2
import numpy as np
import time
import os
import threading

# TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_RUNTIME = False
    except ImportError:
        print("HATA: TensorFlow Lite veya tflite_runtime yuklu degil!")
        print("Kurulum: pip install tflite-runtime")
        exit(1)

# Pi Camera
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("UYARI: Picamera2 bulunamadi. USB kamera kullanilacak.")


# ============ CITYSCAPES SINIF TANIMLARI ============
# 19 sınıflı Cityscapes veri seti
CITYSCAPES_CLASSES = {
    0: ("road", "Yol", (128, 64, 128)),           # Mor
    1: ("sidewalk", "Kaldırım", (244, 35, 232)),   # Pembe
    2: ("building", "Bina", (70, 70, 70)),         # Koyu gri
    3: ("wall", "Duvar", (102, 102, 156)),         # Mor-gri
    4: ("fence", "Çit", (190, 153, 153)),          # Açık mor
    5: ("pole", "Direk", (153, 153, 153)),         # Gri
    6: ("traffic_light", "Trafik lambası", (250, 170, 30)),  # Turuncu
    7: ("traffic_sign", "Trafik işareti", (220, 220, 0)),    # Sarı
    8: ("vegetation", "Bitki örtüsü", (107, 142, 35)),       # Yeşil
    9: ("terrain", "Arazi", (152, 251, 152)),      # Açık yeşil
    10: ("sky", "Gökyüzü", (70, 130, 180)),        # Mavi
    11: ("person", "Insan", (220, 20, 60)),        # Kırmızı
    12: ("rider", "Bisikletli/Motosikletli", (255, 0, 0)),   # Kırmızı
    13: ("car", "Araba", (0, 0, 142)),             # Koyu mavi
    14: ("truck", "Kamyon", (0, 0, 70)),           # Çok koyu mavi
    15: ("bus", "Otobüs", (0, 60, 100)),           # Koyu mavi-gri
    16: ("train", "Tren", (0, 80, 100)),           # Koyu turkuaz
    17: ("motorcycle", "Motosiklet", (0, 0, 230)), # Parlak mavi
    18: ("bicycle", "Bisiklet", (119, 11, 32)),    # Koyu kırmızı
}

# Renk paleti oluştur
def create_color_palette():
    """Cityscapes renk paleti oluşturur"""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (_, _, color) in CITYSCAPES_CLASSES.items():
        # BGR formatına çevir (OpenCV için)
        palette[class_id] = (color[2], color[1], color[0])
    return palette

COLOR_PALETTE = create_color_palette()


# ============ PI CAMERA SINIFI ============
class PiCameraReader:
    """Raspberry Pi Camera Module v3 için threaded reader"""
    
    def __init__(self, camera_num=0, width=1280, height=720):
        self.width = width
        self.height = height
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("Picamera2 kurulu degil!")
        
        try:
            self.picam2 = Picamera2(camera_num)
            config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"},
                buffer_count=2
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            # Autofocus ayarları
            try:
                from libcamera import controls
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfSpeed": controls.AfSpeedEnum.Fast
                })
                print("[OK] Autofocus: Surekli mod aktif")
            except Exception as e:
                print(f"[UYARI] Autofocus ayarlanamadi: {e}")
            
            time.sleep(1.0)
            self.latest_frame = self.picam2.capture_array()
            self.running = True
            
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print(f"[OK] Pi Camera {camera_num} baslatildi ({width}x{height})")
            
        except Exception as e:
            print(f"[HATA] Pi Camera baslatma hatasi: {e}")
            raise
    
    def _update(self):
        """Arka planda sürekli kare yakala"""
        while self.running:
            try:
                frame = self.picam2.capture_array()
                with self.lock:
                    self.latest_frame = frame
            except Exception as e:
                print(f"Kare yakalama hatasi: {e}")
                time.sleep(0.01)
    
    def read(self):
        """En son kareyi döndür"""
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
        print("Pi Camera kapatildi.")
    
    def isOpened(self):
        return self.running


class USBCameraReader:
    """Fallback: USB/IP kamera için reader"""
    
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera acilamadi: {src}")
        
        print(f"[OK] USB Kamera baslatildi")
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    def isOpened(self):
        return self.cap.isOpened()


# ============ DEEPLAB SEGMENTASYON SINIFI ============
class DeepLabSegmenter:
    """DeepLabV3 TFLite modeli ile semantic segmentation"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: TFLite model dosyasının yolu
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadi: {model_path}")
        
        print(f"[*] Model yukleniyor: {model_path}")
        
        # Interpreter oluştur
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Girdi/çıktı detayları
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model boyutlarını al
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"[OK] Model yuklendi")
        print(f"    Girdi boyutu: {self.input_width}x{self.input_height}")
        print(f"    Girdi tipi: {self.input_dtype}")
        print(f"    Cikti boyutu: {self.output_details[0]['shape']}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Görüntüyü model girdisi için hazırla
        
        Args:
            frame: BGR formatında OpenCV görüntüsü
        
        Returns:
            Model için hazırlanmış tensor
        """
        # RGB'ye çevir
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Model boyutuna yeniden boyutlandır
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        
        # Normalize et (gerekiyorsa)
        if self.input_dtype == np.float32:
            # Float model için normalize
            normalized = resized.astype(np.float32) / 255.0
        elif self.input_dtype == np.uint8:
            # Uint8 model için olduğu gibi kullan
            normalized = resized.astype(np.uint8)
        else:
            normalized = resized.astype(self.input_dtype)
        
        # Batch boyutu ekle
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Semantic segmentation tahminini yap
        
        Args:
            frame: BGR formatında OpenCV görüntüsü
        
        Returns:
            Orijinal boyuttaki segmentation mask (H x W)
        """
        original_height, original_width = frame.shape[:2]
        
        # Ön işleme
        input_tensor = self.preprocess(frame)
        
        # İnference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Sonucu al
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Çıktı formatını işle
        if len(output.shape) == 4:
            # (1, H, W, num_classes) formatı - argmax al
            if output.shape[-1] > 1:
                mask = np.argmax(output[0], axis=-1)
            else:
                mask = output[0, :, :, 0]
        elif len(output.shape) == 3:
            # (1, H, W) formatı
            mask = output[0]
        else:
            mask = output
        
        # Orijinal boyuta yeniden boyutlandır
        mask = mask.astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_width, original_height), 
                                   interpolation=cv2.INTER_NEAREST)
        
        return mask_resized
    
    def visualize(self, frame: np.ndarray, mask: np.ndarray, 
                  alpha: float = 0.5) -> np.ndarray:
        """
        Segmentasyon sonucunu görselleştir
        
        Args:
            frame: Orijinal BGR görüntü
            mask: Segmentasyon maskesi
            alpha: Overlay şeffaflığı (0-1)
        
        Returns:
            Overlay uygulanmış görüntü
        """
        # Renk maskesi oluştur
        colored_mask = COLOR_PALETTE[mask]
        
        # Overlay
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay


def draw_legend(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Görüntüde bulunan sınıfların açıklamalarını ekle
    
    Args:
        frame: Görüntü
        mask: Segmentasyon maskesi
    
    Returns:
        Legend eklenmiş görüntü
    """
    # Görüntüde bulunan benzersiz sınıfları bul
    unique_classes = np.unique(mask)
    
    # Her sınıf için yüzde hesapla
    total_pixels = mask.size
    class_percentages = {}
    for class_id in unique_classes:
        if class_id in CITYSCAPES_CLASSES:
            percentage = np.sum(mask == class_id) / total_pixels * 100
            if percentage > 0.5:  # %0.5'ten fazla olanları göster
                class_percentages[class_id] = percentage
    
    # Yüzdeye göre sırala
    sorted_classes = sorted(class_percentages.items(), 
                            key=lambda x: x[1], reverse=True)
    
    # Legend çiz
    y_offset = 60
    for class_id, percentage in sorted_classes[:8]:  # En fazla 8 sınıf göster
        _, turkish_name, color = CITYSCAPES_CLASSES[class_id]
        
        # Renk kutusu
        cv2.rectangle(frame, (10, y_offset - 15), (30, y_offset + 5), 
                     (color[2], color[1], color[0]), -1)
        cv2.rectangle(frame, (10, y_offset - 15), (30, y_offset + 5), 
                     (255, 255, 255), 1)
        
        # Sınıf adı ve yüzde
        text = f"{turkish_name}: %{percentage:.1f}"
        cv2.putText(frame, text, (40, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (40, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        y_offset += 30
    
    return frame


def draw_info(frame: np.ndarray, fps: float, inference_time: float, 
              paused: bool = False) -> np.ndarray:
    """Bilgi metinlerini ekle"""
    h, w = frame.shape[:2]
    
    # Üst bilgi çubuğu
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    
    # FPS ve inference süresi
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Duraklatma durumu
    if paused:
        cv2.putText(frame, "DURAKLATILDI", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Alt bilgi çubuğu
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Q: Cikis | S: Kaydet | P: Duraklat", 
               (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("  DeepLabV3 Semantic Segmentation - Pi Camera Test")
    print("=" * 60)
    
    # Model yolu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "deeplab_cityscapes.tflite")
    
    # Model dosyasını kontrol et
    if not os.path.exists(model_path):
        print(f"\n[HATA] Model dosyasi bulunamadi!")
        print(f"       Beklenen konum: {model_path}")
        print(f"\nLutfen modeli su konuma kopyalayin:")
        print(f"       {model_path}")
        return
    
    # Segmenter oluştur
    try:
        segmenter = DeepLabSegmenter(model_path)
    except Exception as e:
        print(f"[HATA] Model yuklenemedi: {e}")
        return
    
    # Kamera başlat
    print("\n[*] Kamera baslatiliyor...")
    try:
        if PICAMERA_AVAILABLE:
            camera = PiCameraReader(camera_num=0, width=1280, height=720)
        else:
            camera = USBCameraReader(src=0, width=1280, height=720)
    except Exception as e:
        print(f"[HATA] Kamera baslatilamadi: {e}")
        return
    
    print("\n[OK] Sistem hazir! Pencereyi kapatmak icin 'q' tusuna basin.")
    print("-" * 60)
    
    # Değişkenler
    fps = 0
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    last_mask = None
    screenshot_counter = 0
    
    # Ana döngü
    try:
        while True:
            loop_start = time.time()
            
            # Kare oku
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Kare okunamadi!")
                time.sleep(0.1)
                continue
            
            # BGR'ye çevir (picamera RGB verir)
            if PICAMERA_AVAILABLE:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Segmentasyon (duraklı değilse)
            if not paused:
                inference_start = time.time()
                mask = segmenter.predict(frame)
                inference_time = time.time() - inference_start
                last_mask = mask
            else:
                mask = last_mask if last_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
                inference_time = 0
            
            # Görselleştirme
            result = segmenter.visualize(frame, mask, alpha=0.5)
            
            # Legend ve bilgi ekle
            result = draw_legend(result, mask)
            result = draw_info(result, fps, inference_time, paused)
            
            # Göster
            cv2.imshow("DeepLabV3 Segmentation", result)
            
            # FPS hesapla
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nCikis yapiliyor...")
                break
            
            elif key == ord('s') or key == ord('S'):
                screenshot_counter += 1
                filename = f"deeplab_screenshot_{screenshot_counter:04d}.png"
                cv2.imwrite(filename, result)
                print(f"[OK] Ekran goruntusu kaydedildi: {filename}")
            
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                status = "DURAKLATILDI" if paused else "DEVAM EDIYOR"
                print(f"[*] Segmentasyon: {status}")
    
    except KeyboardInterrupt:
        print("\n\nKlavye ile durduruldu.")
    
    finally:
        # Temizlik
        camera.release()
        cv2.destroyAllWindows()
        print("\n[OK] Kaynaklar temizlendi.")


if __name__ == "__main__":
    main()
