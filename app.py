import time
import sys
from services import AudioService, VoiceService, NavigationService

class App:
    def __init__(self):
        print("Sistem başlatılıyor...")
        self.audio_service = AudioService()
        self.nav_service = NavigationService(self.audio_service)
        self.voice_service = VoiceService(self.process_voice_command)
        self.running = True

    def process_voice_command(self, text):
        """Sesli komutları işleyen callback"""
        print(f"Komut işleniyor: {text}")
        
        if "navigasyon" in text and "başlat" in text:
            if not self.nav_service.running:
                self.nav_service.start()
            else:
                print("Navigasyon zaten çalışıyor.")
                
        elif "navigasyon" in text and "dur" in text:
            if self.nav_service.running:
                self.nav_service.stop()
            else:
                print("Navigasyon zaten durmuş.")
                
        elif "kapat" in text or "çıkış" in text:
            print("Uygulama kapatılıyor...")
            self.audio_service.play("DUR")
            self.stop()
            
        elif "sistem" in text and "durum" in text:
            self.audio_service.play("HAZIR")

    def start(self):
        # Servisleri başlat
        self.audio_service.start()
        self.voice_service.start()
        
        self.audio_service.play("HAZIR")
        
        print("Ana döngü başladı. Çıkmak için Ctrl+C basın.")
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.nav_service.stop()
        self.voice_service.stop()
        self.audio_service.stop()
        print("Tüm servisler durduruldu.")
        sys.exit(0)

if __name__ == "__main__":
    app = App()
    app.start()
