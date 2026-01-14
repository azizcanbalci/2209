"""
MOD 5: Sesli AI Sohbet Servisi
Hugging Face Mistral API ile sesli asistan
- Speech-to-Text: Google Speech Recognition
- AI: Mistral-7B via Hugging Face
- Text-to-Speech: gTTS + pygame (mevcut sistemle uyumlu)
"""

import os
import requests
import threading
import time
from dotenv import load_dotenv

# .env dosyasından token'ı yükle
load_dotenv()

# API Ayarları
API_URL = "https://router.huggingface.co/together/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Sistem promptu - görme engelli kullanıcılar için optimize
SYSTEM_PROMPT = """Sen görme engelli bir kullanıcıya yardım eden Türkçe sesli asistansın.
Kurallar:
- Kısa ve öz cevap ver (1-3 cümle)
- Sadece Türkçe konuş
- Samimi ve yardımsever ol
- Görsel olmayan açıklamalar yap
- Kullanıcının çevresini anlamasına yardım et"""


class VoiceChatService:
    """Sesli AI Sohbet Servisi - Singleton"""
    
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
        
        self.hf_token = None
        self.headers = None
        self.model_ready = False
        self.recognizer = None
        self.microphone = None
        self.conversation_history = []
        self.max_history = 10  # Son 10 mesajı tut
        self.is_listening = False
        self._initialized = True
    
    def init(self):
        """Servisi başlat (lazy loading)"""
        if self.model_ready:
            return True
        
        print("🤖 Sesli AI Sohbet başlatılıyor...")
        
        # Token kontrolü
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        if not self.hf_token:
            print("❌ HUGGINGFACE_TOKEN bulunamadı!")
            print("   .env dosyasına ekleyin: HUGGINGFACE_TOKEN=hf_xxxxx")
            return False
        
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # API bağlantı testi
        if not self._test_connection():
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
        print("✓ Sesli AI Sohbet hazır!")
        return True
    
    def _test_connection(self):
        """Hugging Face API bağlantı testi"""
        try:
            test_payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Merhaba"}],
                "max_tokens": 10
            }
            response = requests.post(API_URL, headers=self.headers, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                print("✓ Hugging Face Mistral bağlandı")
                return True
            else:
                print(f"❌ API Hatası: {response.status_code}")
                print(f"   {response.text[:100]}")
                return False
        except requests.Timeout:
            print("❌ API zaman aşımı")
            return False
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def listen(self, timeout=5, phrase_limit=15):
        """
        Mikrofondan ses al ve yazıya çevir
        Returns: (success, text) tuple
        """
        if not self.model_ready:
            return False, "Servis hazır değil"
        
        import speech_recognition as sr
        
        self.is_listening = True
        
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
            
            self.is_listening = False
            
            # Google Speech Recognition ile çevir
            text = self.recognizer.recognize_google(audio, language="tr-TR")
            return True, text.strip()
            
        except sr.WaitTimeoutError:
            self.is_listening = False
            return False, None  # Timeout - sessizlik
        except sr.UnknownValueError:
            self.is_listening = False
            return False, ""  # Anlaşılamadı
        except sr.RequestError as e:
            self.is_listening = False
            return False, f"İnternet hatası: {e}"
        except Exception as e:
            self.is_listening = False
            return False, f"Hata: {e}"
    
    def ask(self, question):
        """
        Mistral'e soru sor ve cevap al
        Returns: AI cevabı (string)
        """
        if not self.model_ready:
            return "Yapay zeka servisine bağlanılamadı."
        
        if not question or not question.strip():
            return "Soru anlaşılamadı."
        
        try:
            # Konuşma geçmişine ekle
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            
            # Geçmiş çok uzunsa kırp
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            # API isteği hazırla
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.conversation_history)
            
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # API'ye gönder
            response = requests.post(
                API_URL, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"].strip()
                    
                    # Cevabı geçmişe ekle
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    return answer if answer else "Cevap üretilemedi."
                return "Cevap üretilemedi."
            
            elif response.status_code == 503:
                return "Model yükleniyor, lütfen tekrar deneyin."
            else:
                print(f"API Hatası: {response.status_code} - {response.text[:100]}")
                return "Bir hata oluştu, tekrar deneyin."
                
        except requests.Timeout:
            return "Bağlantı zaman aşımına uğradı."
        except Exception as e:
            print(f"Mistral Hatası: {e}")
            return "Üzgünüm, şu anda cevap veremiyorum."
    
    def clear_history(self):
        """Konuşma geçmişini temizle"""
        self.conversation_history = []
        return "Konuşma geçmişi temizlendi."
    
    def is_exit_command(self, text):
        """Çıkış komutu mu kontrol et"""
        if not text:
            return False
        
        exit_commands = [
            'kapat', 'çıkış', 'çık', 'güle güle', 'hoşça kal',
            'bye', 'exit', 'modu kapat', 'sohbeti bitir',
            'görüşürüz', 'sonra görüşürüz'
        ]
        
        text_lower = text.lower()
        return any(cmd in text_lower for cmd in exit_commands)


# Singleton instance
voice_chat = VoiceChatService()


def init():
    """Servisi başlat"""
    return voice_chat.init()


def listen(timeout=5, phrase_limit=15):
    """Dinle ve yazıya çevir"""
    return voice_chat.listen(timeout, phrase_limit)


def ask(question):
    """AI'a soru sor"""
    return voice_chat.ask(question)


def clear_history():
    """Geçmişi temizle"""
    return voice_chat.clear_history()


def is_exit_command(text):
    """Çıkış komutu mu?"""
    return voice_chat.is_exit_command(text)


def is_ready():
    """Servis hazır mı?"""
    return voice_chat.model_ready
