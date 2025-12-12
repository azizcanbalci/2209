# Blind Assist - Gelişmiş Engel Tespit ve Yönlendirme Sistemi

Bu proje, görme engelli bireyler için geliştirilmiş, bilgisayarlı görü (computer vision) tabanlı bir yardımcı asistan prototipidir. **YOLOv11** nesne tespiti, **Canny Kenar Tespiti** ve **Inverse Perspective Mapping (IPM)** tekniklerini birleştirerek çevreyi analiz eder ve kullanıcıya en güvenli yürüme rotasını sesli olarak bildirir.

## 🚀 Özellikler

- **Hibrit Algılama:** YOLOv11 ile nesne tespiti ve Canny Edge Detection ile yol sınırlarının belirlenmesi.
- **Kuş Bakışı Görünüm (BEV):** IPM (Inverse Perspective Mapping) ile kamera görüntüsünün kuş bakışı haritaya dönüştürülmesi.
- **Free-Space Analizi:** Yürünebilir güvenli alanların (Free Space) dinamik olarak hesaplanması.
- **Akıllı Yönlendirme:** Sadece engellere değil, boş alanın genişliğine ve sürekliliğine göre karar veren gelişmiş algoritma.
- **Sesli Geri Bildirim:** Türkçe sesli komutlar ("Sola dön", "Düz git", "Dikkat! Çok yakın engel" vb.).
- **Mesafe Tahmini:** Engellerin uzaklığının tahmini ve renk kodlu uyarı sistemi.

## 🛠️ Kurulum

1.  Gerekli kütüphaneleri yükleyin:

    ```powershell
    pip install -r requirements.txt
    ```

2.  PyTorch ve GPU desteği (Opsiyonel ama önerilir):
    Sistem CPU üzerinde çalışabilir ancak daha yüksek FPS için CUDA destekli PyTorch önerilir.

## ▶️ Çalıştırma

Uygulamayı başlatmak için:

```powershell
python main.py
```

Çıkış yapmak için `q` tuşuna basabilirsiniz.

## 🏗️ Sistem Mimarisi

Sistem `VisionPipeline` sınıfı üzerinden modüler bir yapıda çalışır:

1.  **Görüntü Alımı:** Kameradan kare okunur.
2.  **YOLO Inference:** `ultralytics` kütüphanesi ile engeller (insan, araba, sandalye vb.) tespit edilir.
3.  **Edge Detection:** `Canny` algoritması ile yol kenarları ve yapısal sınırlar belirlenir.
4.  **IPM Dönüşümü:** Görüntü perspektifi kaldırılarak 2D kuş bakışı harita oluşturulur.
5.  **Maske Oluşturma:**
    - Kenarlar kalınlaştırılır.
    - YOLO kutuları BEV düzlemine izdüşürülür.
    - Güvenli alanlar (Free Space) beyaz, engeller siyah olarak maskelenir.
6.  **Yol Planlama:** Maske üzerindeki en geniş ve engelsiz şerit (Sol, Orta, Sağ) seçilir.
7.  **Geri Bildirim:** Karar verilen yön sesli olarak kullanıcıya iletilir.

## 📂 Dosya Yapısı

- `main.py`: Ana uygulama döngüsü, ses sistemi ve görselleştirme.
- `vision_pipeline.py`: Görüntü işleme, IPM, maske oluşturma ve yön bulma mantığı.
- `models/`: YOLO model dosyalarının bulunduğu klasör.
- `audio_cache/`: Oluşturulan ses dosyalarının (MP3) önbelleği.
- `AGENTIC.MD`: Proje geliştirme yol haritası ve teknik dokümantasyon.

## 🔧 Gereksinimler

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy
- gTTS (Google Text-to-Speech)
- Pygame (Ses çalma için)
