import speech_recognition as sr
import time
from .base import BaseService

class VoiceService(BaseService):
    def __init__(self, callback_function):
        super().__init__("VoiceService")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback_function

    def run(self):
        # Arka planda dinleme
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        except Exception as e:
            print(f"Mikrofon hatası: {e}")
            return
            
        while self.running:
            try:
                with self.microphone as source:
                    # print("Dinleniyor...") # Konsolu kirletmemek için kapalı
                    try:
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                        command = self.recognizer.recognize_google(audio, language="tr-TR")
                        print(f"Algılanan: {command}")
                        self.callback(command.lower())
                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        print("Google Speech API hatası")
                    
            except Exception as e:
                # Timeout vb.
                pass
            
            time.sleep(0.1)
