"""
Sesli Komut Servisi
Tüm modlar için merkezi ses tanıma sistemi
"""

import threading
import time

# Ses tanıma için
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[UYARI] speech_recognition yuklu degil!")


class VoiceCommandService:
    """
    Merkezi sesli komut servisi.
    Tüm modlarda kullanılacak ses tanıma sistemi.
    """
    
    # Mod seçim komutları
    MODE_COMMANDS = {
        'navigasyon': 1,
        'metin': 2,
        'tanıma': 3,
        'tanima': 3,  # Türkçe karakter alternatifi
        'arama': 4,
        'sohbet': 5,
        'soru': 6,
        'harita': 7,
    }
    
    # Özel komutlar
    EXIT_COMMANDS = ['çık', 'çıkış', 'cik', 'cikis', 'geri']
    SHUTDOWN_COMMANDS = ['kapat', 'programı kapat', 'sistemi kapat', 'uygulamayı kapat']
    CAPTURE_COMMANDS = ['çek', 'cek', 'fotoğraf', 'fotograf', 'yakala']
    SEARCH_COMMANDS = ['ara', 'bul', 'araştır', 'arastir']
    
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
        
        self.recognizer = None
        self.microphone = None
        self.is_ready = False
        self.is_listening = False
        self._initialized = True
    
    def init(self):
        """Servisi başlat"""
        if self.is_ready:
            return True
        
        if not SR_AVAILABLE:
            print("[HATA] speech_recognition kurulu degil!")
            print("  pip install SpeechRecognition pyaudio")
            return False
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Mikrofon kalibrasyonu
            print("[BEKLE] Mikrofon kalibre ediliyor...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            self.is_ready = True
            print("[OK] Sesli komut sistemi hazir!")
            return True
            
        except Exception as e:
            print(f"[HATA] Mikrofon hatasi: {e}")
            return False
    
    def listen(self, timeout=3, phrase_limit=6):
        """
        Mikrofondan ses al ve yazıya çevir.
        HIZLI VERSİYON - Kısa timeout'lar
        
        Returns:
            tuple: (success, text)
            - success: True ise ses tanındı
            - text: Tanınan metin veya hata mesajı
        """
        if not self.is_ready:
            if not self.init():
                return False, "Mikrofon hazir degil"
        
        self.is_listening = True
        
        try:
            with self.microphone as source:
                # Çok kısa kalibrasyon (hız için)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                
                # Dinle - kısa timeout
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
            
            self.is_listening = False
            
            # Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language="tr-TR")
            return True, text.strip().lower()
            
        except sr.WaitTimeoutError:
            self.is_listening = False
            return False, None  # Sessizlik - timeout
        except sr.UnknownValueError:
            self.is_listening = False
            return False, ""  # Ses var ama anlaşılamadı
        except sr.RequestError as e:
            self.is_listening = False
            return False, f"Internet hatasi: {e}"
        except Exception as e:
            self.is_listening = False
            return False, f"Hata: {e}"
    
    def get_mode_from_command(self, text):
        """
        Komut metninden mod numarasını döndür.
        
        Returns:
            int: Mod numarası (1-7) veya None
        """
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        for keyword, mode_num in self.MODE_COMMANDS.items():
            if keyword in text_lower:
                return mode_num
        
        return None
    
    def is_exit_command(self, text):
        """Çıkış komutu mu?"""
        if not text:
            return False
        text_lower = text.lower().strip()
        return any(cmd in text_lower for cmd in self.EXIT_COMMANDS)
    
    def is_shutdown_command(self, text):
        """Program kapatma komutu mu?"""
        if not text:
            return False
        text_lower = text.lower().strip()
        return any(cmd in text_lower for cmd in self.SHUTDOWN_COMMANDS)
    
    def is_capture_command(self, text):
        """Fotoğraf çekme komutu mu?"""
        if not text:
            return False
        text_lower = text.lower().strip()
        return any(cmd in text_lower for cmd in self.CAPTURE_COMMANDS)
    
    def is_search_command(self, text):
        """Arama komutu mu?"""
        if not text:
            return False
        text_lower = text.lower().strip()
        return any(cmd in text_lower for cmd in self.SEARCH_COMMANDS)
    
    def extract_search_target(self, text):
        """
        Arama komutundan hedef nesneyi çıkar.
        Örnek: "sandalye ara" -> "sandalye"
        """
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        # Arama komutlarını kaldır
        for cmd in self.SEARCH_COMMANDS:
            text_lower = text_lower.replace(cmd, '').strip()
        
        # Geriye kalan nesne adı
        if text_lower and len(text_lower) > 1:
            return text_lower
        
        return None
    
    def wait_for_mode_command(self, speak_func=None):
        """
        Mod seçim menüsünde bekle.
        HIZLI VERSİYON
        
        Args:
            speak_func: Sesli geri bildirim fonksiyonu (opsiyonel)
        
        Returns:
            tuple: (command_type, value)
            - ('mode', mode_num): Mod değiştir
            - ('shutdown', None): Programı kapat
            - ('timeout', None): Zaman aşımı
            - ('error', message): Hata
        """
        success, text = self.listen(timeout=5, phrase_limit=5)
        
        if not success:
            if text is None:  # Timeout
                return ('timeout', None)
            return ('error', text)
        
        print(f"[KOMUT] Alindi: {text}")
        
        # Kapatma kontrolü
        if self.is_shutdown_command(text):
            return ('shutdown', None)
        
        # Mod seçimi kontrolü
        mode = self.get_mode_from_command(text)
        if mode:
            return ('mode', mode)
        
        return ('unknown', text)
    
    def wait_for_mode_action(self, current_mode, speak_func=None):
        """
        Mod içinde komut bekle.
        HIZLI VERSİYON
        
        Args:
            current_mode: Mevcut mod numarası
            speak_func: Sesli geri bildirim fonksiyonu
        
        Returns:
            tuple: (action_type, value)
            - ('exit', None): Mod menüsüne dön
            - ('shutdown', None): Programı kapat
            - ('capture', None): Fotoğraf çek (Mod 2, 6)
            - ('search', target): Nesne ara (Mod 4)
            - ('speech', text): Sohbet metni (Mod 5, 6)
            - ('timeout', None): Zaman aşımı
            - ('continue', None): Devam et (Mod 1, 3, 7)
        """
        success, text = self.listen(timeout=4, phrase_limit=8)
        
        if not success:
            if text is None:  # Timeout - sessizlik
                return ('continue', None)
            return ('error', text)
        
        print(f"[KOMUT] Mod {current_mode} - Alindi: {text}")
        
        # Kapatma kontrolü (her modda geçerli)
        if self.is_shutdown_command(text):
            return ('shutdown', None)
        
        # Çıkış kontrolü (her modda geçerli)
        if self.is_exit_command(text):
            return ('exit', None)
        
        # Mod'a özel komutlar
        if current_mode == 2:  # Metin Okuma
            if self.is_capture_command(text):
                return ('capture', None)
        
        elif current_mode == 4:  # Nesne Arama
            if self.is_search_command(text):
                target = self.extract_search_target(text)
                if target:
                    return ('search', target)
                return ('need_target', None)  # Hedef lazım
            # Sadece nesne adı söylenmiş olabilir
            return ('set_target', text)
        
        elif current_mode == 5:  # Sesli Sohbet
            return ('speech', text)
        
        elif current_mode == 6:  # Görsel Soru-Cevap
            if self.is_capture_command(text):
                return ('capture', None)
            return ('speech', text)
        
        # Mod 1, 3, 7 için sadece çıkış/kapat komutları
        return ('continue', None)


# Singleton instance
voice_command = VoiceCommandService()


# Kısayol fonksiyonları
def init():
    """Servisi başlat"""
    return voice_command.init()


def listen(timeout=5, phrase_limit=10):
    """Dinle ve yazıya çevir"""
    return voice_command.listen(timeout, phrase_limit)


def get_mode(text):
    """Komuttan mod numarası al"""
    return voice_command.get_mode_from_command(text)


def is_exit(text):
    """Çıkış komutu mu?"""
    return voice_command.is_exit_command(text)


def is_shutdown(text):
    """Kapatma komutu mu?"""
    return voice_command.is_shutdown_command(text)


def is_capture(text):
    """Çekme komutu mu?"""
    return voice_command.is_capture_command(text)


def wait_for_mode():
    """Mod seçimi bekle"""
    return voice_command.wait_for_mode_command()


def wait_for_action(mode):
    """Mod içi komut bekle"""
    return voice_command.wait_for_mode_action(mode)


def is_ready():
    """Servis hazır mı?"""
    return voice_command.is_ready
