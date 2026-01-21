# Blind Assist - Gelişmiş 7 Modlu Görme Engelli Asistanı

Bu proje, görme engelli bireyler için geliştirilmiş, yapay zeka ve bilgisayarlı görü tabanlı kapsamlı bir yardımcı asistan sistemidir.

## ⚡ v2.0 - Performans Güncellemesi (Ocak 2026)

### Yenilikler
- **🚀 2-3x Daha Hızlı FPS**: PiCamera optimizasyonları (4 buffer, queue=False)
- **⚡ Anlık Sesli Komutlar**: RAM'de önbelleğe alınmış sesler, ~50ms yanıt süresi
- **🎯 Hızlı Navigasyon Tepkisi**: Cooldown'lar %50 azaltıldı
- **🔊 Düşük Latency Audio**: 512 byte buffer ile minimal gecikme
- **📷 Fast Mode Kamera**: 640x480 @ 30+ FPS

### Performans İyileştirmeleri
| Özellik | Önceki | Yeni |
|---------|--------|------|
| Kamera FPS | ~15 FPS | ~30+ FPS |
| Sesli komut gecikmesi | ~1.5 saniye | ~0.5 saniye |
| Yön değişikliği tepkisi | ~2 saniye | ~0.8 saniye |
| YOLO inference | 640px | 416px (daha hızlı) |

## 🚀 7 Mod Sistemi

### MOD 1: Navigasyon
- **YOLOv11** ile nesne tespiti
- **Radar Navigasyon** sistemi ile yön komutları
- **Kuş Bakışı Görünüm (BEV)** ile alan haritalaması
- Sesli yönlendirme: "Sola dön", "Düz git", "Dikkat!"

### MOD 2: Metin Okuma (OCR)
- **PaddleOCR** ile Türkçe metin tanıma
- Sayı ve özel karakterleri koruma
- Manuel tetikleme (SPACE tuşu)
- Türkçe karakter düzeltmeleri

### MOD 3: Nesne Tanıma
- Çevredeki nesnelerin detaylı tanımlanması
- Mesafe tahmini ile yakınlık bilgisi
- Türkçe nesne isimlendirmesi

### MOD 4: Nesne Arama
- Belirli bir nesneyi arama
- Bulunan nesnenin konumu ve uzaklığı
- Sesli yönlendirme ile hedefe ulaşım

### MOD 5: Sesli AI Sohbet
- **Mistral-7B** yapay zeka sohbet
- Sesli komut girişi (mikrofon)
- Türkçe konuşma tanıma ve sentezi

### MOD 6: Görsel Soru-Cevap
- **Gemini 2.5 Flash** görsel analiz
- Fotoğraf hakkında soru sorma
- Detaylı görsel açıklamalar

### MOD 7: 3D Haritalama (SLAM)
- **Monocular Visual SLAM** ile 3D haritalama
- ORB özellik çıkarımı ve eşleştirme
- Essential Matrix ve Triangulation
- PLY formatında harita kaydetme/yükleme
- Kuş bakışı harita görselleştirmesi

## 🛠️ Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# .env dosyasını oluşturun ve API anahtarlarını ekleyin
# HUGGINGFACE_TOKEN=your_token
# GEMINI_API_KEY=your_key
```

## ▶️ Çalıştırma

### Sesli Kontrol Modu (Varsayılan - Önerilen)
```bash
python3 main.py
```

### Klavye Kontrol Modu
```bash
python3 main.py --keyboard
```

### Sesli Mod Komutları
- "navigasyon" → MOD 1
- "metin" → MOD 2
- "tanıma" → MOD 3
- "arama" → MOD 4
- "sohbet" → MOD 5
- "soru" → MOD 6
- "harita" → MOD 7
- "çık" → Mod menüsüne dön
- "kapat" → Programı kapat

### Klavye Kontrolleri
- `1-7`: Mod değiştirme
- `SPACE`: Moda göre tetikleme (OCR okuma, SLAM kaydetme, soru sorma)
- `q`: Çıkış

### MOD 7 Özel Kontrolleri
- `SPACE`: Haritayı kaydet
- `L`: Harita yükle
- `R`: Haritayı sıfırla
- `I`: İstatistikleri göster

## 📂 Dosya Yapısı

```
├── main.py                 # Ana uygulama
├── vision_pipeline.py      # YOLO ve görüntü işleme
├── radar_navigation.py     # Radar navigasyon sistemi
├── navigation_map.py       # Navigasyon haritalaması
├── services/
│   ├── ocr_reader.py       # PaddleOCR Türkçe OCR
│   ├── object_describer.py # Nesne tanımlayıcı
│   ├── object_searcher.py  # Nesne arama
│   ├── voice_chat.py       # Mistral sesli sohbet
│   ├── voice_command.py    # Sesli komut sistemi
│   ├── speech_service.py   # TTS servisi
│   ├── image_qa.py         # Gemini görsel soru-cevap
│   └── slam_mapper.py      # 3D SLAM haritalama
├── models/                 # YOLO model dosyaları
├── audio_cache/            # Ses dosyaları önbelleği
└── maps/                   # Kaydedilen SLAM haritaları
```

## 🔧 Gereksinimler

- Raspberry Pi 5 (önerilen) veya x86 bilgisayar
- Python 3.10+
- PiCamera v3 veya USB kamera
- Mikrofon (sesli komutlar için)
- OpenCV, NumPy, SciPy
- Ultralytics (YOLOv11)
- PaddleOCR, PaddlePaddle
- gTTS, Pygame
- SpeechRecognition, PyAudio
- google-generativeai (Gemini)
- plyfile (PLY formatı)
