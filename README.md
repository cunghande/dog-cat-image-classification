# Phân Loại Hình Ảnh Chó Và Mèo Bằng Mạng Nơ-ron Tích Chập (CNN)

Dự án này sử dụng mạng nơ-ron tích chập (**Convolutional Neural Network - CNN**) để giải quyết bài toán phân loại ảnh nhị phân (Binary Image Classification): nhận diện và phân biệt giữa **Chó (Dogs)** và **Mèo (Cats)**.

---

## 📌 Bộ Dữ Liệu (Dataset)

Dự án sử dụng bộ dữ liệu kinh điển **Dogs vs. Cats** từ Kaggle:
- **Nguồn tải:** [Kaggle - Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

### Cấu trúc thư mục dữ liệu yêu cầu:
Để mô hình có thể tự động nạp dữ liệu thông qua `ImageDataGenerator`, hãy sắp xếp thư mục dữ liệu theo cấu trúc sau:

```text
CNN-phanloaichomeo/
│
├── dataset/
│   ├── training_set/      # 8.000 ảnh (4.000 mèo, 4.000 chó)
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/          # 2.000 ảnh (1.000 mèo, 1.000 chó)
│       ├── cats/
│       └── dogs/
│
├── CNN_phan_loai_cho_meo.ipynb # Notebook báo cáo khoa học (chạy sẵn 100% cell & biểu đồ)
├── CNN_img.py             # Mã nguồn huấn luyện mô hình CNN
├── predict.py             # Script dự đoán hình ảnh Chó/Mèo
├── download_dataset.py    # Script tự động tải và tổ chức dataset
├── run_train.bat          # File click chạy nhanh huấn luyện
├── run_predict.bat        # File click chạy nhanh dự đoán ảnh mẫu
├── sample_cat.jpg         # Ảnh mèo mẫu để thử nghiệm
├── sample_dog.jpg         # Ảnh chó mẫu để thử nghiệm
├── bon1.jpg               # Ảnh thử nghiệm bổ sung
├── bon2.jpg               # Ảnh thử nghiệm bổ sung
└── README.md
```

---

## 🧠 Kiến Trúc Mô Hình CNN

Mô hình được xây dựng theo kiến trúc mạng tích chập tuần tự (`Sequential`):

1. **Lớp Đầu vào (Input Layer):** Kích thước ảnh chuẩn hóa `(64, 64, 3)`.
2. **Lớp Tích chập 1 (Convolutional Layer 1):** 32 bộ lọc (filters) kích thước `3x3`, hàm kích hoạt `ReLU`.
3. **Lớp Gom nhóm 1 (MaxPooling Layer 1):** Kích thước cửa sổ `2x2`.
4. **Lớp Tích chập 2 (Convolutional Layer 2):** 32 bộ lọc kích thước `3x3`, hàm kích hoạt `ReLU`.
5. **Lớp Gom nhóm 2 (MaxPooling Layer 2):** Kích thước cửa sổ `2x2`.
6. **Lớp Làm phẳng (Flatten Layer):** Chuyển đổi ma trận đặc trưng 2D thành vector 1D.
7. **Lớp Kết nối đầy đủ (Dense Layer):** 128 nơ-ron, hàm kích hoạt `ReLU`.
8. **Lớp Đầu ra (Output Layer):** 1 nơ-ron với hàm kích hoạt `Sigmoid` (0 tương ứng với Mèo, 1 tương ứng với Chó).

- **Thuật toán tối ưu (Optimizer):** `Adam`
- **Hàm mất mát (Loss function):** `binary_crossentropy`
- **Chỉ số đánh giá:** `accuracy` (độ chính xác)

---

## 📓 Báo Cáo Thực Nghiệm Bằng Jupyter Notebook (Khuyên dùng cho Báo Cáo / Đồ Án)

File **[`CNN_phan_loai_cho_meo.ipynb`](CNN_phan_loai_cho_meo.ipynb)** đã được **chạy sẵn 100% các cell (Run All Cells)**, chứa đầy đủ kết quả đầu ra, đồ thị độ phân giải cao và bảng số liệu đánh giá chuyên sâu:

1. **Khám phá & Trực quan hóa Dữ liệu (EDA):** Thống kê số lượng, biểu đồ phân bố lớp, lưới ảnh mẫu Chó/Mèo với độ phân giải gốc.
2. **Tiền xử lý & Tăng cường Dữ liệu (Data Augmentation):** Minh họa trực quan các phép biến đổi ảnh (Zoom, Shear, Flip) trên ảnh thực tế.
3. **Kiến trúc Mạng CNN:** Chi tiết các tầng nơ-ron, tham số tính toán (`813,217` tham số).
4. **Huấn luyện Mô hình:** Theo dõi hàm mất mát và độ chính xác theo từng epoch với `EarlyStopping`.
5. **Đường cong Học tập (Learning Curves):** Biểu đồ Loss & Accuracy qua các Epoch (Train vs Validation).
6. **Đánh giá Chuyên sâu trên 2.000 Ảnh Test:**
   - **Ma trận nhầm lẫn (Confusion Matrix):** Heatmap trực quan chi tiết TP, TN, FP, FN.
   - **Báo cáo Phân loại (Classification Report):** Bảng và biểu đồ cột so sánh Precision, Recall, F1-Score cho từng lớp.
   - **Đường cong ROC & Điểm AUC:** Năng lực phân biệt nhị phân đạt **AUC = 0.8502**.
7. **Phân tích Sai số Thực tế (Failure Analysis):** Lưới 16 ảnh test ngẫu nhiên kèm độ tin cậy và phân tích nguyên nhân các ca đoán nhầm.
8. **Thử nghiệm Thực tế:** Dự đoán trên ảnh mẫu ngoài dự án (`sample_cat.jpg`, `sample_dog.jpg`, `bon1.jpg`, `bon2.jpg`).

---

## ⚙️ Yêu Cầu Cài Đặt

### Cài đặt thư viện cần thiết
```bash
pip install tensorflow pillow scipy
```
*(Khuyến nghị Python 3.10 – 3.12 để tương thích tốt nhất với TensorFlow: `py -3.12 -m pip install tensorflow pillow scipy`)*.

---

## 🚀 Hướng Dẫn Sử Dụng Chi Tiết

### Bước 1: Tải bộ dữ liệu (Dataset)
Nếu bạn chưa tải dataset, hãy chạy script tải tự động:

```bash
py -3.12 download_dataset.py
# hoặc:
python download_dataset.py
```
Script sẽ tự động tải file zip từ Microsoft CDN, lọc ảnh lỗi và sắp xếp vào đúng 2 thư mục `dataset/training_set` và `dataset/test_set`.

---

### Bước 2: Huấn luyện mô hình (Training Model)

Bạn có thể chạy bằng một trong hai cách:

- **Cách 1 (Dòng lệnh Terminal / PowerShell):**
  ```bash
  py -3.12 CNN_img.py
  # hoặc:
  python CNN_img.py
  ```
- **Cách 2 (Click chuột):**
  Nhấp đúp chuột vào file **`run_train.bat`**.

Quá trình huấn luyện mặc định chạy 25 epochs. Khi hoàn tất, mô hình được lưu lại thành:
- `cat_dog_classifier.h5`
- `cat_dog_classifier.keras`

> 💡 **Mẹo:** Nếu muốn chạy thử nhanh để kiểm tra (chỉ mất ~2 phút), bạn hãy mở file `CNN_img.py` sửa dòng `epochs=25` thành `epochs=3` hoặc `epochs=5`.

---

### Bước 3: Chạy dự đoán phân loại ảnh (Prediction)

Sau khi huấn luyện xong, bạn có thể kiểm tra xem một bức ảnh bất kỳ là Chó hay Mèo:

#### 1. Dùng script `predict.py` qua dòng lệnh
```bash
# Phân loại ảnh mèo mẫu:
py -3.12 predict.py sample_cat.jpg

# Phân loại ảnh chó mẫu:
py -3.12 predict.py sample_dog.jpg

# Phân loại ảnh bất kỳ của bạn:
py -3.12 predict.py duong_dan_den_anh.jpg
```

**Mẫu kết quả hiển thị:**
```text
========================================
KẾT QUẢ DỰ ĐOÁN:
-> Con vật trong ảnh là: CHÓ (Dog)
-> Độ tin cậy (Confidence): 94.20%
========================================
```

#### 2. Dùng file click chạy nhanh
Nhấp đúp vào file **`run_predict.bat`** để chương trình tự động kiểm tra ảnh mẫu.

#### 3. Nhúng code dự đoán vào ứng dụng khác
```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Tải mô hình đã lưu
model = load_model('cat_dog_classifier.keras')

# Đọc và tiền xử lý ảnh
test_image = image.load_img('duong_dan_den_anh.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image / 255.0, axis=0)

# Dự đoán
result = model.predict(test_image)

if result[0][0] > 0.5:
    print(f"Dự đoán: Chó (Độ tin cậy: {result[0][0] * 100:.2f}%)")
else:
    print(f"Dự đoán: Mèo (Độ tin cậy: {(1 - result[0][0]) * 100:.2f}%)")
```

---

## 💡 Lưu Ý
- Quá trình huấn luyện trên CPU máy tính mất khoảng **15 – 25 phút** cho toàn bộ 25 epochs (~30 giây mỗi epoch).
- Để tăng tốc độ huấn luyện hơn nữa, bạn có thể chạy trên máy có GPU NVIDIA (CUDA) hoặc dùng Google Colab.
