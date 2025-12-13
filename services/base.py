import threading

class BaseService:
    def __init__(self, name):
        self.name = name
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
            print(f"[{self.name}] Başlatıldı.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"[{self.name}] Durduruldu.")

    def run(self):
        pass
