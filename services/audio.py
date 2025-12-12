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
            
        # Pygame mixer başlat
        try:
            pygame.mixer.init()
        except:
            print("Ses sistemi başlatılamadı!")

        # Önceden tanımlı komutları oluştur
        self._create_default_audio_files()

    def _create_default_audio_files(self):
        commands = {
            "SOL": "sola dön",
            "DÜZ": "düz git",
            "SAG": "sağa dön", 
            "DUR": "dur",
            "HAZIR": "Sistem hazır. Komut bekleniyor.",
            "YAKIN": "Dikkat! Çok yakın engel",
            "UZAK": "Yol açık",
            "NAV_BASLA": "Navigasyon başlatılıyor",
            "NAV_DUR": "Navigasyon durduruldu",
            "ANLASILMADI": "Komut anlaşılamadı",
            "HATA": "Bir hata oluştu"
        }
        
        for key, text in commands.items():
            self.generate_audio(key, text)

    def generate_audio(self, key, text):
        filepath = os.path.join(self.audio_dir, f"{key}.mp3")
        if not os.path.exists(filepath):
            try:
                tts = gTTS(text=text, lang='tr')
                tts.save(filepath)
            except Exception as e:
                print(f"Ses oluşturma hatası ({key}): {e}")

    def play(self, command_key):
        """Kuyruğa ses komutu ekler"""
        self.queue.put(command_key)

    def play_immediate(self, command_key):
        """Kuyruğu temizler ve hemen çalar"""
        with self.queue.mutex:
            self.queue.queue.clear()
        self.queue.put(command_key)

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
                print(f"Ses çalma döngüsü hatası: {e}")
