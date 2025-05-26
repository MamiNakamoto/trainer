import os
import subprocess
from pathlib import Path
import shutil

def create_training_dirs():
    # Ana klasörleri oluştur
    dirs = ["/content/runs", "/content/yolov5/runs/train", "/content/yolov5/runs/detect"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # YOLOv5 klasörünü kontrol et
    if not Path("/content/yolov5").exists():
        print("[HATA] YOLOv5 klasörü bulunamadı!")
        print("Lütfen şu komutu çalıştırın:")
        print("git clone https://github.com/ultralytics/yolov5.git")
        return False
    return True

def train_yolov5():
    print("\n[1] YOLOv5 eğitimi başlatılıyor...")
    
    # Klasörleri kontrol et ve oluştur
    if not create_training_dirs():
        return
    
    # Eğitim öncesi klasör yapısını kontrol et
    print("\nKlasör yapısı kontrol ediliyor...")
    print(f"YOLOv5 klasörü: {'✅ Mevcut' if Path('/content/yolov5').exists() else '❌ Eksik'}")
    print(f"data.yaml dosyası: {'✅ Mevcut' if Path('/content/data.yaml').exists() else '❌ Eksik'}")
    print(f"yolov5/runs/train klasörü: {'✅ Mevcut' if Path('/content/yolov5/runs/train').exists() else '❌ Eksik'}")
    
    command = [
        "python", "/content/yolov5/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "50",
        "--data", "/content/data.yaml",
        "--weights", "/content/yolov5s.pt"
    ]
    
    try:
        print("\nEğitim başlatılıyor...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Çıktıyı gerçek zamanlı göster
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Hata varsa göster
        stderr = process.stderr.read()
        if stderr:
            print(f"\n[HATA] Eğitim sırasında hata oluştu:\n{stderr}")
            return
            
        if process.returncode != 0:
            print(f"\n[HATA] Eğitim başarısız oldu. Çıkış kodu: {process.returncode}")
            return
            
        print("\n[1] ✅ Eğitim başarıyla tamamlandı!")
        
        # Eğitim sonrası klasör yapısını kontrol et
        print("\nEğitim sonrası kontrol:")
        exp_folders = list(Path("/content/yolov5/runs/train").glob("exp*"))
        if exp_folders:
            latest_exp = max(exp_folders, key=os.path.getmtime)
            print(f"Son eğitim klasörü: {latest_exp.name}")
            weights_path = latest_exp / "weights/best.pt"
            print(f"Ağırlık dosyası: {'✅ Mevcut' if weights_path.exists() else '❌ Eksik'}")
        else:
            print("❌ Eğitim klasörü bulunamadı!")
            
    except Exception as e:
        print(f"\n[HATA] Beklenmeyen bir hata oluştu: {str(e)}")

def update_yolov5():
    print("\n[2] Yeni verilerle modeli güncelleme başlatılıyor...")
    
    # Önceki eğitim ağırlıklarını kontrol et
    runs_path = Path("/content/yolov5/runs/train")
    
    # Önceki eğitim sonuçlarını kontrol et
    exp_list = sorted(runs_path.glob("exp*"), key=os.path.getmtime)
    if not exp_list:
        print("[HATA] Önceki eğitim sonuçları bulunamadı. Önce ilk eğitimi yapmalısın.")
        return
    
    latest_exp = exp_list[-1]
    best_weights = latest_exp / "weights/best.pt"
    
    if not best_weights.exists():
        print(f"[HATA] Önceki eğitim ağırlıkları bulunamadı: {best_weights}")
        return

    print(f"\nÖnceki eğitim sonucu kullanılıyor: {latest_exp.name}")
    print(f"Ağırlık dosyası: {best_weights}")

    # Yeni eğitim için parametreler
    command = [
        "python", "/content/yolov5/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "20",  # Güncelleme için daha az epoch
        "--data", "/content/data.yaml",
        "--weights", str(best_weights),  # Önceki eğitimin ağırlıklarını kullan
        "--name", f"update_{len(exp_list)}",  # Yeni bir isim ver
        "--exist-ok",  # Aynı isimli klasör varsa üzerine yaz
        "--project", "/content/yolov5/runs/train"  # Proje klasörünü belirt
    ]

    try:
        print("\nGüncelleme başlatılıyor...")
        # Komutu çalıştır ve çıktıyı gerçek zamanlı göster
        process = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        
        # Çıktıyı göster
        if process.stdout:
            print(process.stdout)
        
        # Hata varsa göster
        if process.stderr:
            print(f"\n[HATA] Güncelleme sırasında hata oluştu:\n{process.stderr}")
            return
            
        print("\n[2] ✅ Model başarıyla güncellendi!")
        
        # Güncelleme sonrası kontrol
        print("\nGüncelleme sonrası kontrol:")
        new_exp_folders = list(runs_path.glob("update_*"))
        if new_exp_folders:
            latest_update = max(new_exp_folders, key=os.path.getmtime)
            print(f"Yeni güncelleme klasörü: {latest_update.name}")
            new_weights = latest_update / "weights/best.pt"
            print(f"Yeni ağırlık dosyası: {'✅ Mevcut' if new_weights.exists() else '❌ Eksik'}")
        else:
            print("❌ Güncelleme klasörü bulunamadı!")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[HATA] Güncelleme başarısız oldu. Çıkış kodu: {e.returncode}")
        if e.stdout:
            print(f"Çıktı: {e.stdout}")
        if e.stderr:
            print(f"Hata: {e.stderr}")
    except Exception as e:
        print(f"\n[HATA] Beklenmeyen bir hata oluştu: {str(e)}")

def detect_with_model():
    test_path = input("\n[3] Test etmek istediğin resim veya video dosyasının yolunu gir: ")
    if not os.path.exists(test_path):
        print("[HATA] Dosya bulunamadı.")
        return

    # Eğitim klasörünü kontrol et
    runs_path = Path("/content/yolov5/runs/train")
    if not runs_path.exists():
        print("[HATA] 'runs/train' klasörü bulunamadı. Önce eğitim yapmalısın.")
        return

    try:
        exp_list = sorted(runs_path.glob("exp*"), key=os.path.getmtime)
        if not exp_list:
            print("[HATA] Eğitim sonuçları bulunamadı. Önce eğitim yapmalısın.")
            return

        latest_exp = exp_list[-1]
        best_weights = latest_exp / "weights/best.pt"

        if not best_weights.exists():
            print(f"[HATA] Eğitim ağırlıkları bulunamadı: {best_weights}")
            return

        print(f"\n[3] En son eğitim sonucu kullanılıyor: {latest_exp.name}")
        print(f"[3] Ağırlık dosyası: {best_weights}")

        command = [
            "python", "/content/yolov5/detect.py",
            "--weights", str(best_weights),
            "--source", test_path,
            "--conf", "0.25"
        ]
        subprocess.run(command)
        print("[3] Test tamamlandı! Sonuçlar '/content/runs/detect/exp/' klasörüne kaydedildi.")

    except Exception as e:
        print(f"[HATA] Test sırasında bir hata oluştu: {str(e)}")

def main_menu():
    while True:
        print("\n🐱 YAPAY ZEKA MENÜSÜ")
        print("1. YOLOv5 Eğitimini Başlat (Sıfırdan)")
        print("2. Modeli Güncelle (Yeni Verilerle)")
        print("3. Test Dosyası ile Kedi Algıla")
        print("4. Çıkış")

        choice = input("\nSeçimini yap (1-4): ")

        if choice == "1":
            train_yolov5()
        elif choice == "2":
            update_yolov5()
        elif choice == "3":
            detect_with_model()
        elif choice == "4":
            print("Çıkılıyor. Görüşmek üzere!")
            break
        else:
            print("Geçersiz seçim. Lütfen 1-4 arası gir.")

if __name__ == "__main__":
    main_menu()

