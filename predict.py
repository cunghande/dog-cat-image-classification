# -*- coding: utf-8 -*-
"""
Script dự đoán ảnh Chó hoặc Mèo bằng mô hình đã huấn luyện (cat_dog_classifier.h5)
Cách dùng:
    python predict.py sample_cat.jpg
    python predict.py sample_dog.jpg
    python predict.py path/to/any_image.jpg
"""

import sys
import os

# Tắt log cảnh báo dư thừa từ C++ của TensorFlow & oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "cat_dog_classifier.keras" if os.path.exists("cat_dog_classifier.keras") else "cat_dog_classifier.h5"

def predict_image(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình '{MODEL_PATH}'!")
        print("Vui lòng chạy 'python CNN_img.py' để huấn luyện mô hình trước.")
        return

    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy file ảnh '{img_path}'!")
        return

    print(f"Đang tải mô hình từ '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    print(f"Đang xử lý ảnh: {img_path}")
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image / 255.0, axis=0)

    print("Đang phân loại...")
    result = model.predict(test_image, verbose=0)
    score = result[0][0]

    print("\n" + "="*40)
    print("KẾT QUẢ DỰ ĐOÁN:")
    if score > 0.5:
        confidence = score * 100
        print(f"-> Con vật trong ảnh là: CHÓ (Dog)")
        print(f"-> Độ tin cậy (Confidence): {confidence:.2f}%")
    else:
        confidence = (1 - score) * 100
        print(f"-> Con vật trong ảnh là: MÈO (Cat)")
        print(f"-> Độ tin cậy (Confidence): {confidence:.2f}%")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Mặc định thử ảnh mẫu nếu có
        if os.path.exists("sample_cat.jpg"):
            img_path = "sample_cat.jpg"
        elif os.path.exists("sample_dog.jpg"):
            img_path = "sample_dog.jpg"
        else:
            print("Cách sử dụng: python predict.py <duong_dan_den_anh>")
            sys.exit(1)

    predict_image(img_path)
