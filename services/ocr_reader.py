"""
MOD 2: PaddleOCR tabanlı Metin Okuma Servisi
Türkçe için optimize edilmiş
"""
import cv2
import os
import threading
import re
import numpy as np

# Log mesajlarını kapat
os.environ["PADDLEOCR_LOG_LEVEL"] = "ERROR"

# =============================================
# TÜRKÇE KARAKTER DÜZELTMELERİ
# =============================================
TURKISH_CHAR_MAP = {
    # Yaygın OCR hataları -> Türkçe karşılıkları
    'ý': 'ı', 'Ý': 'I',
    'þ': 'ş', 'Þ': 'Ş',
    'ð': 'ğ', 'Ð': 'Ğ',
    'â': 'a', 'î': 'i', 'û': 'u',
    '|': 'I',
    # NOT: '0' ve '1' dönüşümü kaldırıldı - sayıları bozuyordu
    '@': 'a', '$': 's',
    '€': 'e', '£': 'L',
}

# Harf benzerlik haritası (OCR karıştırmaları)
SIMILAR_CHARS = {
    'i': ['ı', 'l', '1', 'I', '|'],
    'ı': ['i', 'l', '1', 'I', '|'],
    'o': ['0', 'O', 'ö', 'Ö'],
    'ö': ['o', '0', 'O'],
    'u': ['ü', 'U', 'Ü'],
    'ü': ['u', 'U'],
    'c': ['ç', 'C', 'Ç'],
    'ç': ['c', 'C'],
    's': ['ş', 'S', 'Ş', '$'],
    'ş': ['s', 'S', '$'],
    'g': ['ğ', 'G', 'Ğ', '9'],
    'ğ': ['g', 'G', '9'],
}

# Türkçe yaygın kelimeler sözlüğü
TURKISH_DICTIONARY = {
    # Selamlaşma
    'merhaba', 'selam', 'günaydın', 'iyi', 'günler', 'akşamlar', 'geceler',
    'hoşgeldiniz', 'hoşçakal', 'görüşürüz',
    
    # Temel kelimeler
    'evet', 'hayır', 'tamam', 'lütfen', 'teşekkür', 'teşekkürler', 'rica',
    'özür', 'pardon', 'affedersiniz',
    
    # Yönler ve konumlar
    'sağ', 'sol', 'düz', 'ileri', 'geri', 'yukarı', 'aşağı', 'ön', 'arka',
    'üst', 'alt', 'yan', 'karşı', 'köşe',
    
    # Uyarılar
    'dikkat', 'tehlike', 'yasak', 'dur', 'geç', 'bekle', 'gir', 'girme',
    'çık', 'çıkış', 'giriş', 'acil', 'yangın', 'kaçış',
    
    # Mekanlar
    'kapı', 'pencere', 'merdiven', 'asansör', 'tuvalet', 'wc', 'banyo',
    'mutfak', 'salon', 'oda', 'koridor', 'hol', 'bahçe', 'balkon',
    
    # Ulaşım
    'otobüs', 'metro', 'tramvay', 'taksi', 'durak', 'istasyon', 'terminal',
    'havalimanı', 'otogar', 'tren', 'uçak', 'vapur', 'feribot',
    
    # Sayılar yazıyla
    'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on',
    'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan', 'yüz',
    
    # Günler
    'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar',
    
    # Aylar
    'ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran',
    'temmuz', 'ağustos', 'eylül', 'ekim', 'kasım', 'aralık',
    
    # Sık kullanılanlar
    've', 'veya', 'ile', 'için', 'gibi', 'kadar', 'sonra', 'önce',
    'şimdi', 'bugün', 'yarın', 'dün', 'her', 'hiç', 'çok', 'az',
    
    # Türkiye ile ilgili
    'türkiye', 'türk', 'türkçe', 'istanbul', 'ankara', 'izmir',
    
    # Mağaza/işletme
    'market', 'mağaza', 'dükkan', 'restoran', 'kafe', 'cafe', 'eczane',
    'hastane', 'okul', 'banka', 'atm', 'ptt', 'kargo',
    
    # Eylemler
    'aç', 'kapat', 'başla', 'bitir', 'gel', 'git', 'al', 'ver', 'yap',
    'oku', 'yaz', 'dinle', 'konuş', 'bak', 'gör', 'duy', 'hisset',
}

def levenshtein_distance(s1, s2):
    """İki string arasındaki Levenshtein mesafesini hesapla"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_closest_word(word, max_distance=2):
    """Sözlükte en yakın kelimeyi bul"""
    word_lower = word.lower()
    
    # Sayıları atla
    if word.isdigit():
        return None
    
    # Tam eşleşme
    if word_lower in TURKISH_DICTIONARY:
        return word_lower
    
    # Çok kısa kelimeleri atla
    if len(word_lower) < 3:
        return None
    
    # En yakın kelimeyi bul
    best_match = None
    best_distance = max_distance + 1
    
    for dict_word in TURKISH_DICTIONARY:
        # Uzunluk farkı çok fazlaysa atla
        if abs(len(dict_word) - len(word_lower)) > max_distance:
            continue
            
        distance = levenshtein_distance(word_lower, dict_word)
        if distance < best_distance:
            best_distance = distance
            best_match = dict_word
    
    if best_match and best_distance <= max_distance:
        return best_match
    return None

# Özel kelime düzeltmeleri (OCR sık karıştırdığı kelimeler)
DIRECT_CORRECTIONS = {
    # Kelimeleri olduğu gibi koru (Levenshtein hatalı eşleşmelerini önle)
    'nolu': 'nolu', 'no': 'no', 'numara': 'numara', 'numarasi': 'numarası',
    'kat': 'kat', 'metre': 'metre', 'km': 'km', 'cm': 'cm', 'mm': 'mm',
    
    # 1->ı, 0->o dönüşümleri
    'c1k1s': 'çıkış', 'cikis': 'çıkış', 'c1kis': 'çıkış', 'cik1s': 'çıkış',
    'g1r1s': 'giriş', 'giris': 'giriş', 'g1ris': 'giriş', 'gir1s': 'giriş',
    'd1kkat': 'dikkat', 'dikkat': 'dikkat',
    'teh1ike': 'tehlike', 'tehlike': 'tehlike', 'tehl1ke': 'tehlike',
    'ac1l': 'acil', 'acil': 'acil',
    'yang1n': 'yangın', 'yangin': 'yangın',
    'kac1s': 'kaçış', 'kacis': 'kaçış',
    'asans0r': 'asansör', 'asansor': 'asansör',
    '0tobus': 'otobüs', 'otobus': 'otobüs',
    'tuva1et': 'tuvalet', 'tuvaiet': 'tuvalet',
    'merd1ven': 'merdiven', 'merdiven': 'merdiven',
    'kap1': 'kapı', 'kapi': 'kapı',
    'kap1s1': 'kapısı', 'kapisi': 'kapısı', 'kapis1': 'kapısı',
    'pencere': 'pencere',
    'kor1dor': 'koridor', 'koridor': 'koridor',
    'ba1kon': 'balkon', 'baikon': 'balkon',
    'metr0': 'metro', 'metro': 'metro', 'metre': 'metre',
    'tramvay': 'tramvay',
    'durak': 'durak', 'durag1': 'durağı', 'duragi': 'durağı',
    'taks1': 'taksi', 'taksi': 'taksi',
    '1stasyon': 'istasyon', 'istasyon': 'istasyon',
    'hava1iman1': 'havalimanı', 'havalimani': 'havalimanı',
    'turkiye': 'türkiye', 'turk1ye': 'türkiye', 'türkiye': 'türkiye',
    'turkce': 'türkçe', 'türkçe': 'türkçe',
    'gunaydin': 'günaydın', 'gunayd1n': 'günaydın',
    'tesekkur': 'teşekkür', 'tesekk0r': 'teşekkür',
    'tesekkurler': 'teşekkürler',
    'hosgeldiniz': 'hoşgeldiniz', 'hosge1d1n1z': 'hoşgeldiniz',
    'merhaba': 'merhaba', 'merheba': 'merhaba',
    'lutfen': 'lütfen', '1utfen': 'lütfen',
    'guzel': 'güzel', 'guze1': 'güzel',
    'dunya': 'dünya', 'd0nya': 'dünya',
    'gunes': 'güneş',
    'ogrenci': 'öğrenci', '0grenci': 'öğrenci',
    'ogretmen': 'öğretmen', '0gretmen': 'öğretmen',
    'universite': 'üniversite', 'un1vers1te': 'üniversite',
    'hastane': 'hastane', 'hastene': 'hastane',
    'eczane': 'eczane', 'eczene': 'eczane',
    'market': 'market', 'merket': 'market',
    'magaza': 'mağaza', 'ma9aza': 'mağaza',
    'banka': 'banka', 'benka': 'banka',
    'okul': 'okul', '0ku1': 'okul',
    # Ekler
    'var': 'var', 'ver': 'var', 'yok': 'yok',
    'solda': 'solda', 's0lda': 'solda',
    'sagda': 'sağda', 'sa9da': 'sağda',
    'katta': 'katta', 'kette': 'katta',
    'kat': 'kat', 'ket': 'kat',
    'oda': 'oda', '0da': 'oda',
    'yapin': 'yapın', 'yap1n': 'yapın', 'yap': 'yap',
    'girin': 'girin', 'g1r1n': 'girin',
    # Sayılar
    'b1r': 'bir', 'bir': 'bir',
    '1k1': 'iki', 'iki': 'iki',
    'uc': 'üç', 'üc': 'üç',
    'dort': 'dört', 'd0rt': 'dört',
    'bes': 'beş', 'be5': 'beş',
    'alti': 'altı', 'alt1': 'altı',
    'yed1': 'yedi', 'yedi': 'yedi',
    'sek1z': 'sekiz', 'sekiz': 'sekiz',
    'd0kuz': 'dokuz', 'dokuz': 'dokuz',
    '0n': 'on', 'on': 'on',
    # Sıra sayıları
    'b1r1nc1': 'birinci', 'birinci': 'birinci', '1.': '1.',
    '1k1nc1': 'ikinci', 'ikinci': 'ikinci', '2.': '2.',
    'uc1nc1': 'üçüncü', 'ucuncu': 'üçüncü', 'üçüncü': 'üçüncü', '3.': '3.',
    'ucinci': 'üçüncü', 'ücüncü': 'üçüncü',
    'd0rduncu': 'dördüncü', 'dorduncu': 'dördüncü', 'dördüncü': 'dördüncü', '4.': '4.',
    'bes1nc1': 'beşinci', 'besinci': 'beşinci', 'beşinci': 'beşinci', '5.': '5.',
    'altinci': 'altıncı', 'alt1nc1': 'altıncı',
    'yedinci': 'yedinci', 'yed1nc1': 'yedinci',
    'sekizinci': 'sekizinci', 'sek1z1nc1': 'sekizinci',
    'dokuzuncu': 'dokuzuncu', 'd0kuzuncu': 'dokuzuncu',
    'onuncu': 'onuncu', '0nuncu': 'onuncu',
}

def fix_turkish_text(text):
    """Türkçe karakter ve kelime düzeltmeleri uygula - Sayılar korunur"""
    if not text:
        return text
    
    result = text
    
    # 1. Karakter düzeltmeleri (sayılar hariç)
    for wrong, correct in TURKISH_CHAR_MAP.items():
        result = result.replace(wrong, correct)
    
    # 2. Kelime bazlı düzeltme
    words = result.split()
    corrected_words = []
    
    for word in words:
        # Noktalama işaretlerini ayır
        prefix = ''
        suffix = ''
        clean_word = word
        
        # Baştaki noktalama
        while clean_word and not clean_word[0].isalnum():
            prefix += clean_word[0]
            clean_word = clean_word[1:]
        
        # Sondaki noktalama
        while clean_word and not clean_word[-1].isalnum():
            suffix = clean_word[-1] + suffix
            clean_word = clean_word[:-1]
        
        if not clean_word:
            corrected_words.append(word)
            continue
        
        # SAYILARI KORU - sadece rakamlardan oluşuyorsa dokunma
        if clean_word.isdigit():
            corrected_words.append(prefix + clean_word + suffix)
            continue
        
        # Sayı + birim kombinasyonları (örn: 5kg, 100m, 50TL) - koru
        # Ama OCR hatalı kelimeler (c1k1s, g1r1s) düzeltilsin
        word_lower = clean_word.lower()
        
        # Önce direkt düzeltme tablosuna bak (OCR hataları için)
        if word_lower in DIRECT_CORRECTIONS:
            corrected = DIRECT_CORRECTIONS[word_lower]
            # Orijinal büyük/küçük harf yapısını koru
            if clean_word[0].isupper():
                corrected = corrected.capitalize()
            if clean_word.isupper():
                corrected = corrected.upper()
            corrected_words.append(prefix + corrected + suffix)
            continue
        
        # Sayı ile başlayan veya biten kelimeler (5kg, 3.kat) - koru
        if clean_word[0].isdigit() or clean_word[-1].isdigit():
            corrected_words.append(word)
            continue
        
        # Çoğunlukla sayıdan oluşan kelimeler - koru
        digit_count = sum(1 for c in clean_word if c.isdigit())
        if digit_count > len(clean_word) / 2:
            corrected_words.append(word)
            continue
        
        # Sözlükte en yakın kelimeyi bul
        closest = find_closest_word(clean_word, max_distance=2)
        
        if closest:
            # Orijinal büyük/küçük harf yapısını koru
            if clean_word[0].isupper():
                closest = closest.capitalize()
            if clean_word.isupper():
                closest = closest.upper()
            corrected_words.append(prefix + closest + suffix)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def clean_ocr_output(text):
    """OCR çıktısını temizle"""
    if not text:
        return text
    
    # Fazla boşlukları temizle
    text = ' '.join(text.split())
    
    # Anlamsız tekrar eden karakterleri temizle
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Sadece noktalama işaretlerinden oluşan parçaları kaldır
    words = text.split()
    words = [w for w in words if any(c.isalnum() for c in w)]
    
    return ' '.join(words)

class OCRReader:
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
        """Instance değişkenlerini başlat"""
        self.ocr = None
        self.initialized = False
        self.use_new_api = False
        self._init_lock = threading.Lock()
        
    def init(self):
        """Lazy loading - ilk kullanımda yükle (thread-safe)"""
        if self.initialized:
            return True
            
        with self._init_lock:
            if self.initialized:
                return True
                
            try:
                from paddleocr import PaddleOCR
                import logging
                logging.getLogger('ppocr').setLevel(logging.ERROR)
                logging.getLogger('paddle').setLevel(logging.ERROR)
                
                # PaddleOCR başlat - minimal parametreler
                try:
                    self.ocr = PaddleOCR(lang='tr')
                except:
                    self.ocr = PaddleOCR(lang='en')  # Türkçe yoksa İngilizce
                    
                self.initialized = True
                self.use_new_api = hasattr(self.ocr, 'predict')
                
                print("[OK] PaddleOCR yuklendi")
                return True
                
            except ImportError:
                print("[HATA] PaddleOCR yuklu degil! pip install paddlepaddle paddleocr")
                return False
            except Exception as e:
                print(f"[HATA] PaddleOCR hatasi: {e}")
                return False

    def preprocess(self, frame):
        """Görüntü ön işleme - Türkçe OCR için optimize"""
        # Boyut kontrolü - çok küçükse büyüt
        h, w = frame.shape[:2]
        if w < 640:
            scale = 640 / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Gri tonlama
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Kontrast artır (CLAHE) - Türkçe karakterler için daha agresif
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Keskinleştirme
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Parlaklık ve kontrast ayarı
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Adaptif eşikleme (opsiyonel - metin daha belirgin)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                              cv2.THRESH_BINARY, 11, 2)
        
        # BGR'ye geri çevir (PaddleOCR BGR bekliyor)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read(self, frame, use_preprocess=True):
        """
        OCR ile metin oku
        frame: BGR numpy array (OpenCV formatı)
        use_preprocess: Ön işleme uygula
        Returns: Okunan metin string veya None
        """
        if not self.initialized:
            if not self.init():
                return None
        
        try:
            # İşlenmiş veya orijinal görüntü
            img = self.preprocess(frame) if use_preprocess else frame
            
            texts = []
            
            # Yeni API (predict) dene - parametresiz
            if self.use_new_api:
                try:
                    result = self.ocr.predict(img)
                    texts = self._parse_predict_result(result)
                except Exception as e:
                    print(f"predict API hatası: {e}")
                    self.use_new_api = False
            
            # Eski API (ocr) kullan - parametresiz
            if not self.use_new_api:
                try:
                    result = self.ocr.ocr(img)  # cls parametresi KALDIRILDI
                    texts = self._parse_ocr_result(result)
                except Exception as e:
                    # Son çare: sadece predict
                    print(f"ocr API hatası: {e}")
                    try:
                        result = self.ocr.predict(img)
                        texts = self._parse_predict_result(result)
                    except:
                        pass
            
            if texts:
                final_text = " ".join(texts)
                # Temizle ve Türkçe düzeltmeleri uygula
                final_text = clean_ocr_output(final_text)
                final_text = fix_turkish_text(final_text)
                
                if final_text and len(final_text.strip()) > 0:
                    print(f"[OCR] Sonuc: {final_text}")
                    return final_text
            return None
            
        except Exception as e:
            print(f"OCR hatası: {e}")
            return None

    def _parse_predict_result(self, result):
        """Yeni predict API sonucunu işle"""
        texts = []
        if not result:
            return texts
            
        for item in result:
            if isinstance(item, dict):
                # Yeni format: {'rec_texts': [...], 'rec_scores': [...]}
                rec_texts = item.get('rec_texts', [])
                rec_scores = item.get('rec_scores', [])
                
                for i, text in enumerate(rec_texts):
                    text = text.strip()
                    conf = rec_scores[i] if i < len(rec_scores) else 0.5
                    
                    if self._is_valid_text(text, conf):
                        texts.append(text)
                        print(f"  [TEXT] '{text}' (%{conf*100:.0f})")
                        
            elif isinstance(item, (list, tuple)):
                # Alternatif format
                for line in item:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text_info = line[1] if len(line) > 1 else line[0]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0]).strip()
                            conf = float(text_info[1]) if len(text_info) > 1 else 0.5
                            if self._is_valid_text(text, conf):
                                texts.append(text)
                                print(f"  [TEXT] '{text}' (%{conf*100:.0f})")
        return texts

    def _parse_ocr_result(self, result):
        """Eski ocr API sonucunu işle"""
        texts = []
        if not result:
            return texts
            
        # result formatı: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)], ...]
        for page in result:
            if not page:
                continue
            for line in page:
                if not line or len(line) < 2:
                    continue
                    
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0]).strip()
                    conf = float(text_info[1])
                    
                    if self._is_valid_text(text, conf):
                        texts.append(text)
                        print(f"  [TEXT] '{text}' (%{conf*100:.0f})")
                        
        return texts

    def _is_valid_text(self, text, confidence):
        """Metnin geçerli olup olmadığını kontrol et - Türkçe için optimize"""
        # Boş veya çok kısa
        if not text or len(text) < 1:
            return False
        
        # Düşük güvenilirlik - Türkçe için daha toleranslı
        if confidence < 0.2:
            return False
        
        # Sadece sembol
        if all(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' for c in text):
            return False
            
        return True


# Singleton instance
ocr_reader = OCRReader()


def read_text(frame):
    """Kısa erişim fonksiyonu"""
    return ocr_reader.read(frame)


# Test
if __name__ == '__main__':
    print("PaddleOCR Test")
    print("=" * 40)
    
    # Kamera testi
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera açılamadı, IP kamera deneniyor...")
        cap = cv2.VideoCapture("http://172.18.160.61:8080/video")
    
    print("Kameradan görüntü alınıyor...")
    ret, frame = cap.read()
    
    if ret:
        print(f"Frame boyutu: {frame.shape}")
        print("OCR çalıştırılıyor...")
        
        text = read_text(frame)
        
        if text:
            print(f"\n{'='*40}")
            print(f"SONUÇ: {text}")
            print(f"{'='*40}")
        else:
            print("Metin bulunamadı")
        
        # Görüntüyü göster
        cv2.imshow("OCR Test", frame)
        cv2.waitKey(0)
    else:
        print("Kare alınamadı!")
    
    cap.release()
    cv2.destroyAllWindows()
