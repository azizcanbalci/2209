"""
Basit Sesli Asistan
- Kullanıcının sesini dinler (Speech-to-Text)
- Hugging Face Mistral API'ye gönderir
- Cevabı sesli okur (Text-to-Speech)
"""

import speech_recognition as sr
import os
import requests
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import tempfile

# .env dosyasından token'ı yükle
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Hugging Face API URL (Together AI Router - OpenAI uyumlu)
API_URL = "https://router.huggingface.co/together/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

class SesliAsistan:
    def __init__(self, hf_token):
        print("Sesli Asistan başlatılıyor...")
        
        self.hf_token = hf_token
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        self.conversation_history = []
        
        # Hugging Face bağlantı testi
        try:
            test_payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            test_response = requests.post(API_URL, headers=self.headers, json=test_payload)
            if test_response.status_code == 200:
                print("✓ Hugging Face Mistral bağlandı.")
                self.model_ready = True
            else:
                print(f"✗ Hugging Face hatası: {test_response.status_code} - {test_response.text[:100]}")
                self.model_ready = False
        except Exception as e:
            print(f"✗ Bağlantı hatası: {e}")
            self.model_ready = False

        # Ses Tanıma (Speech-to-Text)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("✓ Mikrofon hazır.")
        
        # Ses Sentezleme (Text-to-Speech) - gTTS + pygame
        pygame.mixer.init()
        print("✓ TTS hazır.")

    def dinle(self):
        """Mikrofondan ses al ve yazıya çevir."""
        with self.microphone as source:
            print("\n🎤 Dinliyorum... (Konuşabilirsiniz)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                print("⏳ Ses işleniyor...")
                
                text = self.recognizer.recognize_google(audio, language="tr-TR")
                print(f"👤 Siz: {text}")
                return text
                
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                print("❓ Anlaşılamadı.")
                return ""
            except sr.RequestError:
                print("❌ İnternet bağlantısı yok!")
                return ""

    def soyle(self, text):
        """Metni Türkçe sesli oku (gTTS + pygame)."""
        print(f"🤖 Asistan: {text}")
        try:
            # gTTS ile ses oluştur
            tts = gTTS(text=text, lang='tr')
            
            # Geçici dosyaya kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # pygame ile çal
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Ses bitene kadar bekle
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Geçici dosyayı sil
            os.unlink(temp_file)
            
        except Exception as e:
            print(f"TTS Hatası: {e}")

    def mistral_sor(self, soru):
        """Mistral'e soru sor ve cevap al."""
        if not self.model_ready:
            return "Yapay zeka servisine bağlanılamadı."
        
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Sen yardımsever bir Türkçe sesli asistansın. Kısa ve öz cevap ver (1-3 cümle). Sadece Türkçe cevap ver."},
                    {"role": "user", "content": soru}
                ],
                "max_tokens": 150
            }
            
            response = requests.post(API_URL, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["message"]["content"]
                    return generated_text.strip() if generated_text else "Cevap üretilemedi."
                return "Cevap üretilemedi."
            else:
                print(f"API Hatası: {response.status_code} - {response.text}")
                return "Bir hata oluştu."
                
        except Exception as e:
            print(f"Mistral Hatası: {e}")
            return "Üzgünüm, şu anda cevap veremiyorum."

    def calistir(self):
        """Ana döngü - sürekli dinle ve cevapla."""
        print("\n" + "="*40)
        print("🎙️  SESLİ ASİSTAN HAZIR")
        print("="*40)
        print("Çıkmak için 'kapat' veya 'çıkış' deyin.\n")
        
        self.soyle("Merhaba! Size nasıl yardımcı olabilirim?")
        
        while True:
            kullanici_metni = self.dinle()
            
            if kullanici_metni is None:
                continue
            
            if kullanici_metni == "":
                self.soyle("Sizi anlayamadım, tekrar eder misiniz?")
                continue
            
            # Çıkış komutu kontrolü
            cikis_komutlari = ['kapat', 'çıkış', 'çık', 'güle güle', 'hoşça kal', 'bye', 'exit']
            if any(komut in kullanici_metni.lower() for komut in cikis_komutlari):
                self.soyle("Görüşmek üzere, hoşça kalın!")
                break
            
            # Mistral'e sor
            cevap = self.mistral_sor(kullanici_metni)
            
            # Cevabı seslendir
            self.soyle(cevap)


def main():
    if not HUGGINGFACE_TOKEN:
        print("HATA: HUGGINGFACE_TOKEN bulunamadı!")
        print("Lütfen .env dosyasına token'ınızı ekleyin:")
        print('HUGGINGFACE_TOKEN=hf_xxxxx')
        return
    
    asistan = SesliAsistan(HUGGINGFACE_TOKEN)
    asistan.calistir()


if __name__ == "__main__":
    main()
