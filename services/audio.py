import os
import time
import threading
from queue import Queue
from gtts import gTTS
import pygame
from .base import BaseService

class AudioService(BaseService):
    def __init__(self):
        super().__init__("AudioService")
        self.queue = Queue()
        self.audio_dir = "audio_cache"
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        
        pygame.mixer.init()
        self.create_default_files()

    def create_default_files(self):
        commands = {
            "SOL": "sola dön",
            "DÜZ": "düz git",
            "SAG": "sağa dön", 
            "DUR": "dur",
            "HAZIR": "Sistem hazır",
            "YAKIN": "Dikkat! Çok yakın engel",
            "UZAK": "Yol açık",
            "NAV_BASLA": "Navigasyon başlatıldı",
            "NAV_DUR": "Navigasyon durduruldu",
            "HATA": "Bir hata oluştu"
        }
        
        for key, text in commands.items():
            filepath = os.path.join(self.audio_dir, f"{key}.mp3")
            if not os.path.exists(filepath):
                print(f"Ses dosyası oluşturuluyor: {text}")
                try:
                    tts = gTTS(text=text, lang='tr')
                    tts.save(filepath)
                except Exception as e:
                    print(f"TTS Hatası: {e}")

    def play(self, command):
        self.queue.put(command)

    def play_immediate(self, command):
        # Kuyruğu temizle ve hemen çal
        with self.queue.mutex:
            self.queue.queue.clear()
        self.queue.put(command)

    def run(self):
        while self.running:
            try:
                if not self.queue.empty():
                    command = self.queue.get(timeout=0.1)
                    filepath = os.path.join(self.audio_dir, f"{command}.mp3")
                    if os.path.exists(filepath):
                        pygame.mixer.music.load(filepath)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and self.running:
                            time.sleep(0.1)
                    self.queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Audio Loop Hatası: {e}")
