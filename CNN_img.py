# -*- coding: utf-8 -*-
"""
Mô hình mạng nơ-ron tích chập (CNN) phân loại nhị phân: Chó vs Mèo
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. XÂY DỰNG KIẾN TRÚC MÔ HÌNH CNN
# ==========================================
classifier = Sequential([
    Input(shape=(64, 64, 3)),
    # Lớp tích chập 1: 32 bộ lọc kích thước 3x3
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Lớp tích chập 2: 32 bộ lọc kích thước 3x3
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Làm phẳng dữ liệu thành vector 1 chiều
    Flatten(),

    # Lớp kết nối đầy đủ (Fully Connected Layer)
    Dense(units=128, activation='relu'),

    # Lớp đầu ra: 1 nơ-ron với hàm kích hoạt sigmoid (0: Mèo, 1: Chó)
    Dense(units=1, activation='sigmoid')
])

# Biên dịch mô hình với thuật toán tối ưu Adam và hàm mất mát Binary Crossentropy
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Hiển thị tóm tắt kiến trúc mạng
classifier.summary()

# ==========================================
# 2. TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)
# ==========================================
# Tăng cường dữ liệu cho tập huấn luyện nhằm tránh overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Chuẩn hóa giá trị điểm ảnh về đoạn [0, 1]
    shear_range=0.2,        # Biến dạng ảnh
    zoom_range=0.2,         # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True    # Lật ảnh theo chiều ngang
)

# Với tập kiểm thử, chỉ thực hiện chuẩn hóa pixel
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Đọc dữ liệu ảnh từ các thư mục con
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH
# ==========================================
# Lưu ý: fit() tự động xác định số bước (steps) dựa trên kích thước tập dữ liệu
classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set
)

# ==========================================
# 4. LƯU MÔ HÌNH SAU KHI HUẤN LUYỆN
# ==========================================
classifier.save('cat_dog_classifier.h5')
classifier.save('cat_dog_classifier.keras')
print("Đã hoàn thành huấn luyện và lưu mô hình vào 'cat_dog_classifier.h5' & 'cat_dog_classifier.keras'!")
