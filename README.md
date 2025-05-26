# YOLOv5 Nesne Tespiti Projesi

Bu proje, YOLOv5 kullanarak nesne tespiti yapan bir model içermektedir.

## Proje Yapısı

- `main.py`: Ana uygulama dosyası
- `data.yaml`: Veri seti konfigürasyonu
- `dataset/`: Eğitim ve test verileri
- `best.pt`: En iyi model ağırlıkları
- `last.pt`: Son eğitim modeli
- `yolov5s.pt`: Başlangıç modeli

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Modeli çalıştırın:
```bash
python main.py
```

## Kullanım

Model, görüntülerde nesne tespiti yapmak için eğitilmiştir. Detaylı kullanım talimatları için `main.py` dosyasına bakınız. 