import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import threading
import time
import queue

class VoiceChatAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_active = False
        self.listening_thread = None
        
        # Gemini Ayarları
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.chat = self.model.start_chat(history=[])
            print("Gemini AI bağlandı.")
        except Exception as e:
            print(f"Gemini bağlantı hatası: {e}")
            self.model = None

        # Ses Tanıma (STT)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Ses Sentezleme (TTS)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Ses kuyruğu (Thread güvenliği için)
        self.audio_queue = queue.Queue()

    def speak(self, text):
        """Metni sesli okur."""
        print(f"Asistan: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Hatası: {e}")

    def listen_and_process(self):
        """Tek bir dinleme-cevaplama döngüsü."""
        with self.microphone as source:
            print("Dinleniyor... (Konuşabilirsiniz)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # 5 saniye bekle, konuşma başlarsa 10 saniye dinle
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("Ses işleniyor...")
                text = self.recognizer.recognize_google(audio, language="tr-TR")
                print(f"Kullanıcı: {text}")
                
                # Gemini'ye sor
                response_text = self.get_gemini_response(text)
                
                # Cevabı seslendir
                self.speak(response_text)
                
            except sr.WaitTimeoutError:
                pass # Kimse konuşmadı
            except sr.UnknownValueError:
                print("Anlaşılamadı.")
                self.speak("Sizi anlayamadım, tekrar eder misiniz?")
            except sr.RequestError:
                print("İnternet hatası.")
                self.speak("İnternet bağlantısında sorun var.")
            except Exception as e:
                print(f"Hata: {e}")

    def get_gemini_response(self, text):
        if not self.model:
            return "Yapay zeka servisine bağlanılamadı."
        
        try:
            prompt = f"Sen görme engelli bir kullanıcıya yardım eden bir sesli asistansın. Kullanıcı sana şunu sordu: '{text}'. Lütfen kısa, net ve samimi bir cevap ver."
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini Hatası: {e}")
            return "Üzgünüm, şu anda cevap veremiyorum."

    def start_session(self):
        self.is_active = True
        self.speak("Sohbet modu aktif. Sizi dinliyorum.")

    def stop_session(self):
        self.is_active = False
        self.speak("Sohbet modu kapatılıyor.")

    def process_step(self):
        """Ana döngüden çağrılacak tek adım."""
        if self.is_active:
            self.listen_and_process()
