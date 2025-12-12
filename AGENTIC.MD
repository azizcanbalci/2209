# Proje Dokümantasyonu

## 📄 Project Brief

**Proje Özeti**
Bu proje, YOLO nesne tespiti ile edge detection + free-space analizini birleştirerek görme engelliler için gerçek zamanlı güvenli yol bulma ve sesli yönlendirme sistemi geliştirmeyi amaçlar. Sistem tıpkı otonom araçlardaki gibi perception → free-space → path planning mantığıyla çalışır.

**Amaç**

- Kameradan alınan görüntülerle yürünebilir alanı tespit etmek
- Engelleri YOLO ile algılamak
- Kenar tespiti (Canny) + IPM ile boş alan maskesi üretmek
- Bu iki bilgiyi birleştirerek karar verme mekanizması oluşturmak
- Kullanıcıya sade sesli komutlarla yön bilgisini aktarmak

**Ana Özellikler**

- YOLO ile engel algılama
- Edge + IPM ile free-space çıkarma
- Kuş bakışı alan analizi
- En güvenli koridoru bulma
- Sesli yönlendirme
- Mobil/JT/Jetson uyumluluğu

## 📄 Product Context

**Problemin Tanımı**
Görme engelli kullanıcılar için mevcut çözümler ya çok pahalı (lidar tabanlı) ya da düşük doğruluklu. Kamerayla çalışan gerçek zamanlı bir yol bulma sistemi eksiktir.

**Ürünün Sağladığı Fayda**

- Kullanıcının güvenli şekilde yürümesine yardım eder
- Engel algılama + yol analizi tek pakette
- Düşük donanımlarda bile çalışabilir
- Kask, gözlük veya telefon şeklinde entegre olabilir

**Kullanıcı Deneyimi Hedefleri**

- Tek cümlelik net yönlendirmeler (“Sağa dön”, “Düz ilerle”)
- 20–30 FPS gerçek zamanlı çalışma
- Yanlış alarmı azaltan çift katmanlı algılama (YOLO + Canny)

## 📄 System Patterns

**Sistem Mimarisi**

1. Görüntü Yakalama
2. YOLO İşleme (Obstacle Detection)
3. Edge Detection + IPM (Free-Space Extraction)
4. Mask Birleştirme → ObstacleMap
5. Koridor Hesaplama (Path Planning)
6. Karar Üretme (Rule-Based Planner)
7. Sesli Yönlendirme

**Ana Desenler**

- Perception → Planning pipeline
- Free-space segmentation
- Column scanning corridor detection
- Rule-based decision making

**Kritik İş Akışları**

- YOLO kutularını free-space maskesi üzerine bindirme
- BEV (bird-eye view) üretimi
- En geniş boş alanı bulma
- Yakın engel alarmı

## 📄 Tech Context

**Kullanılan Teknolojiler**

- YOLOv8 / YOLOv11
- OpenCV (Canny, morphology, warpPerspective)
- MiDaS (opsiyonel depth)
- Python
- ONNX Runtime / TensorRT (optimizasyon)

**Teknik Kısıtlar**

- Mobil cihazlarda hız sınırlı
- Depth olmadan yakınlık tahmini hata payı içerir
- IPM için kamera açı kalibrasyonu gerekir

**Geliştirme Araçları**

- Python
- VSCode + Cline
- OpenCV
- ONNX

## 📄 Active Context

**Son Aktif Durum**

- Temel görüntü işleme pipeline'ı (YOLO + Canny + IPM) tamamlandı.
- Hibrit Engel Haritası (Free-Space Mask + YOLO Projeksiyonu) oluşturuldu.
- Gelişmiş yön bulma algoritması (Free-Space tabanlı) entegre edildi.
- Sesli geri bildirim sistemi (gTTS + Pygame) sorunsuz çalışıyor.

**Tamamlanan Görevler**

- [x] YOLO nesne tespiti entegrasyonu
- [x] Canny kenar tespiti
- [x] IPM (Inverse Perspective Mapping) dönüşümü
- [x] BEV (Bird's Eye View) Free-Space Maskesi
- [x] YOLO kutularının BEV haritasına izdüşümü
- [x] Free-Space tabanlı akıllı yön bulma algoritması
- [x] Türkçe sesli komut sistemi

**Sırada Ne Var?**

- Saha testleri ve IPM kalibrasyonunun ince ayarı.
- Performans optimizasyonu (Jetson/Raspberry Pi için).

## 📄 Progress

**Ne İşliyor?**

- Sistem gerçek zamanlı olarak engelleri ve yol sınırlarını tespit ediyor.
- Kuş bakışı harita üzerinde güvenli yürünebilir alanlar (yeşil) görselleştiriliyor.
- En geniş boş koridora göre "SOL", "SAĞ" veya "DÜZ" komutları veriliyor.
- Yakın engellerde "DUR" veya "YAKIN" uyarısı yapılıyor.

**Ne Eksik?**

- Farklı kamera açıları için otomatik kalibrasyon aracı.
- Derinlik (Depth) tahmini için stereo kamera veya MiDaS entegrasyonu (Gelecek sürüm).

**Hatalar / Riskler**

- IPM matrisi sabit değerlerle çalışıyor; kamera açısı değişirse kalibrasyon bozulabilir.
- Çok karmaşık ışık koşullarında Canny kenar tespiti gürültülü olabilir.

**Sonuç**

- Projenin temel prototipi başarıyla tamamlandı. Sistem, belirlenen mimariye uygun olarak Perception -> Mapping -> Planning -> Action döngüsünü gerçekleştiriyor.
