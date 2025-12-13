import time
import sys
from services.audio import AudioService
from services.voice import VoiceService
from services.navigation import NavigationService

class App:
    def __init__(self):
        print("Uygulama başlatılıyor...")
        
        # Servisleri Başlat
        self.audio_service = AudioService()
        self.navigation_service = NavigationService(self.audio_service)
        self.voice_service = VoiceService(self.handle_voice_command)
        
        self.running = True
        
    def handle_voice_command(self, command):
        """Sesli komutları işler"""
        print(f"Algılanan Komut: {command}")
        
        if "başlat" in command or "aktif" in command:
            if not self.navigation_service.running:
                self.navigation_service.start()
                self.audio_service.play("HAZIR") # Veya "Navigasyon Başlatıldı"
            else:
                self.audio_service.play("HAZIR") # Zaten çalışıyor
                
        elif "dur" in command or "bekle" in command:
            if self.navigation_service.running:
                self.navigation_service.stop()
                
        elif "kapat" in command or "çıkış" in command:
            self.stop()
            
    def start(self):
        """Uygulamayı başlatır"""
        # Ses servisini başlat (Dosyaları oluşturur)
        self.audio_service.start()
        
        # Sesli komut dinlemeyi başlat
        self.voice_service.start()
        
        self.audio_service.play("HAZIR")
        print("Sistem Hazır. 'Navigasyon Başlat' diyebilirsiniz.")
        
        # Ana döngü
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Uygulamayı durdurur"""
        print("Uygulama kapatılıyor...")
        self.running = False
        self.navigation_service.stop()
        self.voice_service.stop()
        self.audio_service.stop()
        sys.exit(0)

if __name__ == "__main__":
    app = App()
    app.start()
