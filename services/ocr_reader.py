"""
MOD 2: Tesseract OCR tabanli Metin Okuma Servisi
Raspberry Pi 5 (64-bit Bookworm) icin optimize edilmis
Turkce karakter destegi ile
"""
import cv2
import os
import threading
import re
import numpy as np

# =============================================
# TURKCE KARAKTER DUZELTMELERI
# =============================================
TURKISH_CHAR_MAP = {
    # Yaygin OCR hatalari -> Turkce karsiliklari
    'ý': 'ı', 'Ý': 'I',
    'þ': 'ş', 'Þ': 'Ş',
    'ð': 'ğ', 'Ð': 'Ğ',
    'â': 'a', 'î': 'i', 'û': 'u',
    '|': 'I',
    '@': 'a', '$': 's',
    '€': 'e', '£': 'L',
}

# Turkce yaygin kelimeler sozlugu
TURKISH_DICTIONARY = {
    # Selamlasma
    'merhaba', 'selam', 'günaydin', 'iyi', 'günler', 'aksamlar', 'geceler',
    'hosgeldiniz', 'hoscakal', 'görüsürüz',
    
    # Temel kelimeler
    'evet', 'hayir', 'tamam', 'lütfen', 'tesekkür', 'tesekkürler', 'rica',
    'özür', 'pardon', 'affedersiniz',
    
    # Yonler ve konumlar
    'sag', 'sol', 'düz', 'ileri', 'geri', 'yukari', 'asagi', 'ön', 'arka',
    'üst', 'alt', 'yan', 'karsi', 'köse',
    
    # Uyarilar
    'dikkat', 'tehlike', 'yasak', 'dur', 'gec', 'bekle', 'gir', 'girme',
    'cik', 'cikis', 'giris', 'acil', 'yangin', 'kacis',
    
    # Mekanlar
    'kapi', 'pencere', 'merdiven', 'asansör', 'tuvalet', 'wc', 'banyo',
    'mutfak', 'salon', 'oda', 'koridor', 'hol', 'bahce', 'balkon',
    
    # Ulasim
    'otobüs', 'metro', 'tramvay', 'taksi', 'durak', 'istasyon', 'terminal',
    'havalimani', 'otogar', 'tren', 'ucak', 'vapur', 'feribot',
    
    # Sayilar yaziyla
    'bir', 'iki', 'üc', 'dört', 'bes', 'alti', 'yedi', 'sekiz', 'dokuz', 'on',
    'yirmi', 'otuz', 'kirk', 'elli', 'altmis', 'yetmis', 'seksen', 'doksan', 'yüz',
    
    # Günler
    'pazartesi', 'sali', 'carsamba', 'persembe', 'cuma', 'cumartesi', 'pazar',
    
    # Aylar
    'ocak', 'subat', 'mart', 'nisan', 'mayis', 'haziran',
    'temmuz', 'agustos', 'eylül', 'ekim', 'kasim', 'aralik',
    
    # Sik kullanilanlar
    've', 'veya', 'ile', 'icin', 'gibi', 'kadar', 'sonra', 'önce',
    'simdi', 'bugün', 'yarin', 'dün', 'her', 'hic', 'cok', 'az',
    
    # Turkiye ile ilgili
    'türkiye', 'türk', 'türkce', 'istanbul', 'ankara', 'izmir',
    
    # Magaza/isletme
    'market', 'magaza', 'dükkan', 'restoran', 'kafe', 'cafe', 'eczane',
    'hastane', 'okul', 'banka', 'atm', 'ptt', 'kargo',
    
    # Eylemler
    'ac', 'kapat', 'basla', 'bitir', 'gel', 'git', 'al', 'ver', 'yap',
    'oku', 'yaz', 'dinle', 'konus', 'bak', 'gör', 'duy', 'hisset',
}


def fix_turkish_text(text):
    """Turkce karakter duzeltmeleri uygula"""
    if not text:
        return text
    
    result = text
    
    # Karakter duzeltmeleri
    for wrong, correct in TURKISH_CHAR_MAP.items():
        result = result.replace(wrong, correct)
    
    return result


def clean_ocr_output(text):
    """OCR ciktisini temizle"""
    if not text:
        return text
    
    # Fazla bosluklari temizle
    text = ' '.join(text.split())
    
    # Anlamsiz tekrar eden karakterleri temizle
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Sadece noktalama isaretlerinden olusan parcalari kaldir
    words = text.split()
    words = [w for w in words if any(c.isalnum() for c in w)]
    
    return ' '.join(words)


class OCRReader:
    """Tesseract OCR ile metin okuma - Singleton"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Thread-safe singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_vars()
        return cls._instance
    
    def _init_vars(self):
        """Instance degiskenlerini baslat"""
        self.tesseract = None
        self.initialized = False
        self._init_lock = threading.Lock()
        self.tesseract_cmd = None
        
    def init(self):
        """Lazy loading - ilk kullanimda yukle (thread-safe)"""
        if self.initialized:
            return True
            
        with self._init_lock:
            if self.initialized:
                return True
                
            try:
                import pytesseract
                
                # Tesseract binary yolunu ayarla
                # Raspberry Pi / Linux
                if os.path.exists('/usr/bin/tesseract'):
                    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
                # Windows
                elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                elif os.path.exists(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'):
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                
                self.tesseract = pytesseract
                
                # Test - Tesseract calisiyormu?
                try:
                    version = pytesseract.get_tesseract_version()
                    print(f"[OK] Tesseract OCR yuklendi (v{version})")
                except Exception as e:
                    print(f"[UYARI] Tesseract versiyon alinamadi: {e}")
                
                # Turkce dil dosyasi var mi kontrol et
                try:
                    langs = pytesseract.get_languages()
                    if 'tur' in langs:
                        print("[OK] Turkce dil destegi mevcut")
                        self.lang = 'tur+eng'  # Turkce + Ingilizce
                    else:
                        print("[UYARI] Turkce dil dosyasi yok, Ingilizce kullanilacak")
                        print("  Yuklemek icin: sudo apt-get install tesseract-ocr-tur")
                        self.lang = 'eng'
                except:
                    self.lang = 'eng'
                
                self.initialized = True
                return True
                
            except ImportError:
                print("[HATA] pytesseract yuklu degil! pip install pytesseract")
                return False
            except Exception as e:
                print(f"[HATA] Tesseract hatasi: {e}")
                print("  Tesseract yuklu mu? sudo apt-get install tesseract-ocr")
                return False

    def preprocess(self, frame):
        """Goruntu on isleme - OCR icin optimize"""
        # Gri tonlama
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Boyut kontrolu - cok kucukse buyut
        h, w = gray.shape[:2]
        if w < 640:
            scale = 640 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Gurultu azaltma
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Kontrast artir (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Adaptif esikleme (metin icin en iyi sonuc)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morfolojik islemler - gurultu temizleme
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def read(self, frame, use_preprocess=True):
        """
        OCR ile metin oku
        frame: BGR numpy array (OpenCV formati)
        use_preprocess: On isleme uygula
        Returns: Okunan metin string veya None
        """
        if not self.initialized:
            if not self.init():
                return None
        
        try:
            # On isleme
            if use_preprocess:
                img = self.preprocess(frame)
            else:
                # Sadece gri tonlama
                if len(frame.shape) == 3:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    img = frame
            
            # Tesseract konfigurasyonu
            # PSM modlari:
            # 3 = Fully automatic page segmentation (default)
            # 6 = Assume a single uniform block of text
            # 7 = Treat the image as a single text line
            # 11 = Sparse text
            config = f'--oem 3 --psm 6 -l {self.lang}'
            
            # OCR calistir
            text = self.tesseract.image_to_string(img, config=config)
            
            if text:
                # Temizle ve duzelt
                text = clean_ocr_output(text)
                text = fix_turkish_text(text)
                
                if text and len(text.strip()) > 1:
                    print(f"[OCR] Sonuc: {text}")
                    return text.strip()
            
            return None
            
        except Exception as e:
            print(f"[HATA] OCR hatasi: {e}")
            return None

    def read_with_boxes(self, frame):
        """
        OCR ile metin oku ve kutucuklari dondur
        Returns: (text, boxes) tuple
        boxes: [(x, y, w, h, text, conf), ...]
        """
        if not self.initialized:
            if not self.init():
                return None, []
        
        try:
            img = self.preprocess(frame)
            
            # Detayli veri al
            data = self.tesseract.image_to_data(
                img, 
                config=f'--oem 3 --psm 6 -l {self.lang}',
                output_type=self.tesseract.Output.DICT
            )
            
            boxes = []
            full_text = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Dusuk guvenilirlik veya bos metni atla
                if conf < 30 or not text:
                    continue
                
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                boxes.append((x, y, w, h, text, conf))
                full_text.append(text)
                print(f"  [TEXT] '{text}' (%{conf})")
            
            result_text = ' '.join(full_text)
            result_text = clean_ocr_output(result_text)
            result_text = fix_turkish_text(result_text)
            
            return result_text, boxes
            
        except Exception as e:
            print(f"[HATA] OCR boxes hatasi: {e}")
            return None, []

    def draw_boxes(self, frame, boxes):
        """Tespit edilen metin kutucuklarini frame uzerine ciz"""
        result = frame.copy()
        
        for (x, y, w, h, text, conf) in boxes:
            # Yesil kutucuk
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Metin etiketi
            label = f"{text} ({conf}%)"
            cv2.putText(result, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result


# Singleton instance
ocr_reader = OCRReader()


def read_text(frame):
    """Kisa erisim fonksiyonu"""
    return ocr_reader.read(frame)


def read_text_with_boxes(frame):
    """Kutucuklarla birlikte oku"""
    return ocr_reader.read_with_boxes(frame)


# Test
if __name__ == '__main__':
    print("Tesseract OCR Test")
    print("=" * 40)
    
    # Kamera testi
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera acilamadi!")
        exit()
    
    print("Kameradan goruntu aliniyor...")
    ret, frame = cap.read()
    
    if ret:
        print(f"Frame boyutu: {frame.shape}")
        print("OCR calistiriliyor...")
        
        text, boxes = read_text_with_boxes(frame)
        
        if text:
            print(f"\n{'='*40}")
            print(f"SONUC: {text}")
            print(f"{'='*40}")
            
            # Kutucuklari ciz ve goster
            result = ocr_reader.draw_boxes(frame, boxes)
            cv2.imshow("OCR Test", result)
            cv2.waitKey(0)
        else:
            print("Metin bulunamadi")
            cv2.imshow("OCR Test", frame)
            cv2.waitKey(0)
    else:
        print("Kare alinamadi!")
    
    cap.release()
    cv2.destroyAllWindows()
