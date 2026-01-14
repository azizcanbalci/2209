"""
Ses Servisi - TTS (Text-to-Speech) işlemleri
gTTS + pygame kullanarak Türkçe ses sentezi
"""
import os
import time
import threading
import hashlib
from queue import Queue

# Global değişkenler
speech_queue = Queue()
pygame_initialized = False
audio_cache_dir = "audio_cache"


def init_speech():
    """Ses sistemini başlat"""
    global pygame_initialized
    
    try:
        import pygame
        pygame.mixer.init()
        pygame_initialized = True
        
        # Önbellek klasörünü oluştur
        if not os.path.exists(audio_cache_dir):
            os.makedirs(audio_cache_dir)
        
        print("✅ Ses sistemi başlatıldı")
        return True
    except Exception as e:
        print(f"❌ Ses sistemi hatası: {e}")
        return False


def get_cached_audio_path(text):
    """Metin için önbellek dosya yolu döndür"""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return os.path.join(audio_cache_dir, f"{text_hash}.mp3")


def speak_text(text):
    """Metni seslendir (senkron)"""
    global pygame_initialized
    
    if not text:
        return
    
    if not pygame_initialized:
        init_speech()
    
    try:
        import pygame
        from gtts import gTTS
        
        # Önbellekten kontrol et
        cache_path = get_cached_audio_path(text)
        
        if not os.path.exists(cache_path):
            # Yeni ses dosyası oluştur
            tts = gTTS(text=text, lang='tr')
            tts.save(cache_path)
        
        # Ses çal
        pygame.mixer.music.load(cache_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"TTS hatası: {e}")


def speak_text_async(text):
    """Metni seslendir (asenkron)"""
    if text:
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()


def speak_text_temp(text):
    """Geçici dosya ile seslendir (OCR için)"""
    global pygame_initialized
    
    if not text:
        return
    
    if not pygame_initialized:
        init_speech()
    
    try:
        import pygame
        from gtts import gTTS
        
        # Geçici dosya oluştur
        temp_file = os.path.join(audio_cache_dir, f"temp_{int(time.time()*1000)}.mp3")
        
        tts = gTTS(text=text, lang='tr')
        tts.save(temp_file)
        
        # Ses çal
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Geçici dosyayı sil
        try:
            pygame.mixer.music.unload()
            os.remove(temp_file)
        except:
            pass
            
    except Exception as e:
        print(f"TTS hatası: {e}")


def speak_text_temp_async(text):
    """Geçici dosya ile seslendir (asenkron)"""
    if text:
        threading.Thread(target=speak_text_temp, args=(text,), daemon=True).start()


# Navigasyon komutları için önbellek oluştur
NAVIGATION_COMMANDS = {
    "DÜZ": "Düz",
    "SOLA": "Sola",
    "SAĞA": "Sağa",
    "GERİ": "Geri çekil",
    "DUR": "Dur",
    "DUR_SOL": "Dur, sola kaç",
    "DUR_SAG": "Dur, sağa kaç",
    "DUR_GERI": "Dur, geri çekil",
    "ENGEL_SOL": "Engel solda",
    "ENGEL_SAG": "Engel sağda",
    "YAKIN": "Dikkat yakın",
    "YAKIN_SOL": "Yakın, sola git",
    "YAKIN_SAG": "Yakın, sağa git"
}


def precache_navigation_sounds():
    """Navigasyon seslerini önceden oluştur"""
    print("🔊 Navigasyon sesleri yükleniyor...")
    
    for command, text in NAVIGATION_COMMANDS.items():
        cache_path = get_cached_audio_path(command)
        if not os.path.exists(cache_path):
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='tr')
                tts.save(cache_path)
            except:
                pass
    
    print("✅ Navigasyon sesleri hazır")


def speak_navigation_command(command):
    """Navigasyon komutunu seslendir"""
    global pygame_initialized
    
    if not pygame_initialized:
        init_speech()
    
    try:
        import pygame
        
        cache_path = get_cached_audio_path(command)
        
        if os.path.exists(cache_path):
            pygame.mixer.music.load(cache_path)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Nav ses hatası: {e}")


# Test
if __name__ == '__main__':
    init_speech()
    precache_navigation_sounds()
    
    print("Test: Merhaba")
    speak_text("Merhaba, ben görme engelli asistanınızım")
    
    print("Test: Navigasyon")
    speak_navigation_command("DÜZ")
    time.sleep(1)
    speak_navigation_command("SOLA")
