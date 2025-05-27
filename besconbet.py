# Eski modeli yükle
import sys
import torch
import pathlib

sys.path.append('/content/yolov5')
from models.yolo import DetectionModel

pathlib.WindowsPath = pathlib.PosixPath
torch.serialization.add_safe_globals({'DetectionModel': DetectionModel})

model = torch.load('/content/best.pt', map_location='cpu', weights_only=False)

# Yeni, temiz bir .pt dosyası olarak kaydet
torch.save(model, '/content/best_converted.pt')

print("✅ Model dönüştürüldü ve best_converted.pt olarak kaydedildi.")
