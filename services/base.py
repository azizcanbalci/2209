from abc import ABC, abstractmethod
import threading

class BaseService(ABC):
    def __init__(self, name):
        self.name = name
        self.running = False
        self.thread = None

    def start(self):
        """Servisi başlatır (genellikle yeni bir thread içinde)"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_internal, daemon=True)
            self.thread.start()
            print(f"[{self.name}] Servis başlatıldı.")

    def stop(self):
        """Servisi durdurur"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            print(f"[{self.name}] Servis durduruldu.")

    def _run_internal(self):
        """Thread içinde çalışacak sarmalayıcı"""
        try:
            self.run()
        except Exception as e:
            print(f"[{self.name}] Hata: {e}")
            self.running = False

    @abstractmethod
    def run(self):
        """Alt sınıfların implemente etmesi gereken ana döngü"""
        pass
