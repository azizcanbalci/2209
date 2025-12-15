import speech_recognition as sr
import pyttsx3
import requests
import os

class VoiceChatAgent:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.is_active = False
        
        # Hugging Face bağlantı testi
        try:
            test_response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": "Test", "parameters": {"max_new_tokens": 5}}
            )
            if test_response.status_code in [200, 503]:
                print("Hugging Face Mistral bağlandı.")
                self.model_ready = True
            else:
                print(f"Hugging Face hatası: {test_response.status_code}")
                self.model_ready = False
        except Exception as e:
            print(f"Bağlantı hatası: {e}")
            self.model_ready = False

        # Ses Tanıma (STT)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Ses Sentezleme (TTS)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

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
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("Ses işleniyor...")
                text = self.recognizer.recognize_google(audio, language="tr-TR")
                print(f"Kullanıcı: {text}")
                
                # Mistral'e sor
                response_text = self.get_mistral_response(text)
                
                # Cevabı seslendir
                self.speak(response_text)
                
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("Anlaşılamadı.")
                self.speak("Sizi anlayamadım, tekrar eder misiniz?")
            except sr.RequestError:
                print("İnternet hatası.")
                self.speak("İnternet bağlantısında sorun var.")
            except Exception as e:
                print(f"Hata: {e}")

    def get_mistral_response(self, text):
        """Mistral'e soru sor ve cevap al."""
        if not self.model_ready:
            return "Yapay zeka servisine bağlanılamadı."
        
        try:
            prompt = f"""<s>[INST] Sen görme engelli bir kullanıcıya yardım eden bir sesli asistansın. Kısa ve öz cevap ver (1-3 cümle). Türkçe cevap ver.

Kullanıcı: {text} [/INST]"""
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "").strip()
                    return generated_text if generated_text else "Cevap üretilemedi."
                return "Cevap üretilemedi."
            elif response.status_code == 503:
                return "Model yükleniyor, lütfen bekleyin..."
            else:
                print(f"API Hatası: {response.status_code}")
                return "Bir hata oluştu."
                
        except Exception as e:
            print(f"Mistral Hatası: {e}")
