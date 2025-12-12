import speech_recognition as sr
import time
from .base import BaseService

class VoiceService(BaseService):
    def __init__(self, callback_function):
        super().__init__("VoiceService")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback_function
        
        # Arka plan gürültüsünü ayarla
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def run(self):
        print("[VoiceService] Dinleme başladı...")
        while self.running:
            try:
                with self.microphone as source:
                    # Kısa süreli dinleme
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        text = self.recognizer.recognize_google(audio, language="tr-TR")
                        text = text.lower()
                        print(f"[VoiceService] Algılandı: {text}")
                        
                        if self.callback:
                            self.callback(text)
                            
                    except sr.WaitTimeoutError:
                        pass # Konuşma yok, devam et
                    except sr.UnknownValueError:
                        pass # Anlaşılamadı
                    except sr.RequestError:
                        print("Google Speech API hatası")
                        time.sleep(1)
                        
            except Exception as e:
                print(f"[VoiceService] Hata: {e}")
                time.sleep(1)
