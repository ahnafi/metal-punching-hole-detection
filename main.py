import os
import cv2
import numpy as np
import shutil
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import time
from ultralytics import YOLO

app = Flask(__name__)

# Konfigurasi folder penyimpanan
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Batas maksimal 16 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Inisialisasi model YOLO
model = YOLO('model.pt')

def process_image(filepath, filename):
    # Membaca gambar asli
    img = cv2.imread(filepath)
    if img is None:
        return None

    # 1. Mengubah ke bentuk grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Binary Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Opening
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Mengembalikan ke bentuk BGR agar kompatibel dengan YOLO
    processed_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    
    # 4. Pemrosesan Model YOLO (Langsung menggunakan array numpy)
    results = model.predict(source=processed_bgr, save=False, conf=0.25)
    
    detections = []
    # Palet warna unik untuk setiap deteksi
    colors = [
        (0, 255, 0),   # Hijau
        (255, 0, 0),   # Biru
        (0, 0, 255),   # Merah
        (0, 255, 255), # Kuning
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
        (128, 0, 128), # Ungu
        (0, 128, 128)  # Teal
    ]
    
    result = results[0]
    boxes = result.boxes
    names = result.names
    
    for i, box in enumerate(boxes):
        # Mengambil koordinat bounding box
        b = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = names[cls]
        
        num = i + 1
        color = colors[i % len(colors)]
        
        # Menggambar kotak pada processed_bgr
        cv2.rectangle(processed_bgr, (x1, y1), (x2, y2), color, 3)
        # Menggambar kotak pada gambar asli (img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Menulis label (#1 (0.85))
        # label = f"#{num} ({conf:.2f})"
        label = f"#{num}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        
        # Background teks agar tulisan kontras (pada processed_bgr)
        cv2.rectangle(processed_bgr, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
        cv2.putText(processed_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # Background teks agar tulisan kontras (pada img asli)
        cv2.rectangle(img, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # Menyimpan data deteksi untuk dilempar ke template HTML
        detections.append({
            'num': num,
            'conf': conf,
            'class_name': class_name
        })
        
    # Menyimpan hasil akhir ke folder results
    result_filename = "result_" + filename
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, processed_bgr)
    
    # Simpan/timpa gambar asli yang sudah diberi kotak ke static/uploads
    cv2.imwrite(filepath, img)
        
    return result_filename, detections

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Pengecekan apakah ada file yang diupload
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada file yang dipilih.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Tidak ada file yang dipilih.')
            
        if file and allowed_file(file.filename):
            # Mengamankan nama file dan menambahkan timestamp agar unik
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Melakukan pemrosesan gambar
            result_data = process_image(filepath, filename)
            
            if result_data:
                result_filename, detections = result_data
                return render_template('index.html', 
                                       original_image=filename,
                                       result_image=result_filename,
                                       detections=detections)
            else:
                return render_template('index.html', error='Gagal memproses gambar.')
                
        else:
            return render_template('index.html', error='Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.')
    
    return render_template('index.html')

if __name__ == '__main__':
    # Pastikan folder tersedia
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    app.run(debug=True, port=5000)
