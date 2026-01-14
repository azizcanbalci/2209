"""
MOD 6: Görsel Soru-Cevap Servisi (Image Q&A)
Gemini 2.0 Flash ile görüntü analizi ve soru yanıtlama
- Kameradan fotoğraf çeker
- Kullanıcıdan sesli soru alır
- Gemini'ye gönderir ve yanıtı seslendirir
"""

import os
import base64
import tempfile
import threading
import time
from dotenv import load_dotenv

# .env dosyasından API key'i yükle
load_dotenv()

# Gemini API ayarları
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"  # 2.0-flash kota sorunu olduğu için 2.5-flash kullanıyoruz

# Sistem promptu - görme engelli kullanıcılar için optimize
SYSTEM_PROMPT = """Sen görme engelli bir kullanıcıya yardım eden görsel asistansın.
Görevlerin:
- Resimdeki öğeleri net ve anlaşılır şekilde betimle
- Kullanıcının sorduğu soruları resme bakarak yanıtla
- Kısa, öz ve Türkçe cevap ver (2-4 cümle)
- Mekansal bilgileri açıkça belirt (sol, sağ, ön, arka, yakın, uzak)
- Tehlikeli durumları özellikle vurgula
- Renkleri, yazıları ve detayları söyle"""


class ImageQAService:
    """Görsel Soru-Cevap Servisi - Singleton"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.api_key = None
        self.model = None
        self.model_ready = False
        self.recognizer = None
        self.microphone = None
        self.temp_files = []  # Silinecek geçici dosyalar
        self._initialized = True
    
    def init(self):
        """Servisi başlat (lazy loading)"""
        if self.model_ready:
            return True
        
        print("📷 Görsel Soru-Cevap başlatılıyor...")
        
        # API Key kontrolü
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            print("❌ GEMINI_API_KEY bulunamadı!")
            print("   .env dosyasına ekleyin: GEMINI_API_KEY=your_api_key")
            return False
        
        # Gemini SDK'yı yükle
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(MODEL_NAME)
            print(f"✓ Gemini {MODEL_NAME} bağlandı")
        except ImportError:
            print("❌ google-generativeai kütüphanesi yüklü değil!")
            print("   pip install google-generativeai")
            return False
        except Exception as e:
            print(f"❌ Gemini bağlantı hatası: {e}")
            return False
        
        # Speech Recognition başlat
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Mikrofon kalibrasyonu
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("✓ Mikrofon hazır")
        except Exception as e:
            print(f"❌ Mikrofon hatası: {e}")
            return False
        
        self.model_ready = True
        print("✓ Görsel Soru-Cevap hazır!")
        return True
    
    def capture_frame(self, frame):
        """
        OpenCV frame'ini geçici dosyaya kaydet
        Returns: Dosya yolu veya None
        """
        try:
            import cv2
            
            # Geçici dosya oluştur
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.jpg',
                prefix='gemini_capture_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Frame'i kaydet
            cv2.imwrite(temp_path, frame)
            
            # Silinecek listeye ekle
            self.temp_files.append(temp_path)
            
            print(f"📸 Fotoğraf kaydedildi: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"❌ Fotoğraf kaydetme hatası: {e}")
            return None
    
    def listen_question(self, timeout=7, phrase_limit=15):
        """
        Kullanıcıdan sesli soru al
        Returns: (success, text) tuple
        """
        if not self.model_ready:
            return False, "Servis hazır değil"
        
        import speech_recognition as sr
        
        try:
            with self.microphone as source:
                # Ortam gürültüsüne ayarla
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                
                # Dinle
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_limit
                )
            
            # Google Speech Recognition ile çevir
            text = self.recognizer.recognize_google(audio, language="tr-TR")
            return True, text.strip()
            
        except sr.WaitTimeoutError:
            return False, None  # Timeout - sessizlik
        except sr.UnknownValueError:
            return False, ""  # Anlaşılamadı
        except sr.RequestError as e:
            return False, f"İnternet hatası: {e}"
        except Exception as e:
            return False, f"Hata: {e}"
    
    def ask_gemini(self, image_path, question):
        """
        Gemini'ye resim ve soru gönder, yanıt al
        Returns: AI yanıtı (string)
        """
        if not self.model_ready:
            return "Servis hazır değil."
        
        if not image_path or not os.path.exists(image_path):
            return "Fotoğraf bulunamadı."
        
        if not question or not question.strip():
            return "Soru anlaşılamadı."
        
        try:
            import google.generativeai as genai
            from PIL import Image
            
            # Resmi yükle
            image = Image.open(image_path)
            
            # Prompt oluştur
            full_prompt = f"""{SYSTEM_PROMPT}

Kullanıcının sorusu: {question}

Lütfen resmi analiz ederek bu soruyu Türkçe yanıtla."""
            
            # Gemini'ye gönder
            response = self.model.generate_content([full_prompt, image])
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Yanıt alınamadı."
                
        except Exception as e:
            error_str = str(e)
            print(f"❌ Gemini hatası: {e}")
            
            # Kota hatası kontrolü
            if "429" in error_str or "quota" in error_str.lower():
                return "API kotası doldu. Lütfen birkaç dakika bekleyin veya yeni API anahtarı alın."
            elif "403" in error_str or "permission" in error_str.lower():
                return "API erişim izni yok. API anahtarınızı kontrol edin."
            elif "401" in error_str or "invalid" in error_str.lower():
                return "Geçersiz API anahtarı. Lütfen doğru anahtarı girin."
            else:
                return f"Bir hata oluştu. Lütfen tekrar deneyin."
    
    def cleanup(self):
        """Geçici dosyaları temizle"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Silindi: {file_path}")
            except Exception as e:
                print(f"⚠️ Dosya silinemedi: {file_path} - {e}")
        
        self.temp_files = []
    
    def process_query(self, frame, question):
        """
        Tek bir sorgu işle: fotoğraf çek, Gemini'ye sor, temizle
        Returns: AI yanıtı
        """
        # 1. Fotoğrafı kaydet
        image_path = self.capture_frame(frame)
        if not image_path:
            return "Fotoğraf çekilemedi."
        
        # 2. Gemini'ye sor
        answer = self.ask_gemini(image_path, question)
        
        # 3. Temizle
        self.cleanup()
        
        return answer


# Singleton instance
image_qa = ImageQAService()


def init():
    """Servisi başlat"""
    return image_qa.init()


def listen_question(timeout=7, phrase_limit=15):
    """Sesli soru al"""
    return image_qa.listen_question(timeout, phrase_limit)


def process_query(frame, question):
    """Resim + soru ile AI'dan yanıt al"""
    return image_qa.process_query(frame, question)


def ask_gemini(image_path, question):
    """Direkt Gemini'ye sor"""
    return image_qa.ask_gemini(image_path, question)


def cleanup():
    """Geçici dosyaları temizle"""
    image_qa.cleanup()


def is_ready():
    """Servis hazır mı?"""
    return image_qa.model_ready
