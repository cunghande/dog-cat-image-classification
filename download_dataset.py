# -*- coding: utf-8 -*-
"""
Script tự động tải và tổ chức bộ dữ liệu Chó & Mèo (Cats vs Dogs)
Tạo cấu trúc thư mục:
dataset/
  ├── training_set/
  │   ├── cats/ (4000 ảnh)
  │   └── dogs/ (4000 ảnh)
  └── test_set/
      ├── cats/ (1000 ảnh)
      └── dogs/ (1000 ảnh)
"""

import os
import sys
import time
import zipfile
import urllib.request
import shutil

# Ensure UTF-8 output on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_PATH = "kagglecatsanddogs_5340.zip"
BASE_DIR = "dataset"
TRAIN_CATS = os.path.join(BASE_DIR, "training_set", "cats")
TRAIN_DOGS = os.path.join(BASE_DIR, "training_set", "dogs")
TEST_CATS = os.path.join(BASE_DIR, "test_set", "cats")
TEST_DOGS = os.path.join(BASE_DIR, "test_set", "dogs")

NUM_TRAIN_PER_CLASS = 4000
NUM_TEST_PER_CLASS = 1000

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"File zip đã tồn tại: {target_path} (Bỏ qua bước tải)")
        return

    print(f"Bắt đầu tải dataset từ Microsoft CDN...")
    print(f"URL: {url}")
    
    start_time = time.time()
    
    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        elapsed = time.time() - start_time
        if elapsed > 0:
            speed = (downloaded / (1024 * 1024)) / elapsed
        else:
            speed = 0
        if total_size > 0:
            percent = downloaded * 100 / total_size
            total_mb = total_size / (1024 * 1024)
            down_mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rĐang tải: {down_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%) - {speed:.2f} MB/s")
        else:
            down_mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rĐang tải: {down_mb:.1f} MB - {speed:.2f} MB/s")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print("\nTải xuống hoàn tất!")

def is_valid_jpeg(data):
    # Kiểm tra kích thước tối thiểu và header JPEG (\xff\xd8)
    if len(data) < 100:
        return False
    if not (data[0] == 0xFF and data[1] == 0xD8):
        return False
    return True

def extract_and_organize(zip_path):
    print("Đang tạo các thư mục đích...")
    for folder in [TRAIN_CATS, TRAIN_DOGS, TEST_CATS, TEST_DOGS]:
        os.makedirs(folder, exist_ok=True)

    print("Đang giải nén và lọc ảnh vào tập training_set và test_set...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        cat_count = 0
        dog_count = 0
        target_total = NUM_TRAIN_PER_CLASS + NUM_TEST_PER_CLASS

        for file_info in zf.infolist():
            filename = file_info.filename
            
            # Chỉ xử lý file jpg trong PetImages/Cat hoặc PetImages/Dog
            if filename.startswith("PetImages/Cat/") and filename.lower().endswith(".jpg"):
                if cat_count >= target_total:
                    continue
                data = zf.read(file_info)
                if not is_valid_jpeg(data):
                    continue
                
                cat_count += 1
                if cat_count <= NUM_TRAIN_PER_CLASS:
                    dest = os.path.join(TRAIN_CATS, f"cat.{cat_count}.jpg")
                else:
                    dest = os.path.join(TEST_CATS, f"cat.{cat_count - NUM_TRAIN_PER_CLASS}.jpg")
                
                with open(dest, "wb") as f:
                    f.write(data)
                
                if cat_count % 1000 == 0:
                    print(f"Đã xử lý {cat_count}/{target_total} ảnh mèo...")

            elif filename.startswith("PetImages/Dog/") and filename.lower().endswith(".jpg"):
                if dog_count >= target_total:
                    continue
                data = zf.read(file_info)
                if not is_valid_jpeg(data):
                    continue
                
                dog_count += 1
                if dog_count <= NUM_TRAIN_PER_CLASS:
                    dest = os.path.join(TRAIN_DOGS, f"dog.{dog_count}.jpg")
                else:
                    dest = os.path.join(TEST_DOGS, f"dog.{dog_count - NUM_TRAIN_PER_CLASS}.jpg")
                
                with open(dest, "wb") as f:
                    f.write(data)

                if dog_count % 1000 == 0:
                    print(f"Đã xử lý {dog_count}/{target_total} ảnh chó...")

            if cat_count >= target_total and dog_count >= target_total:
                break

    # Trích xuất thêm 2 ảnh mẫu để người dùng thử nghiệm dự đoán ngay
    sample_cat_path = "sample_cat.jpg"
    sample_dog_path = "sample_dog.jpg"
    if os.path.exists(os.path.join(TEST_CATS, "cat.1.jpg")) and not os.path.exists(sample_cat_path):
        shutil.copy(os.path.join(TEST_CATS, "cat.1.jpg"), sample_cat_path)
    if os.path.exists(os.path.join(TEST_DOGS, "dog.1.jpg")) and not os.path.exists(sample_dog_path):
        shutil.copy(os.path.join(TEST_DOGS, "dog.1.jpg"), sample_dog_path)

    print("\n" + "="*50)
    print("HOÀN TẤT THIẾT LẬP DATASET!")
    print(f"- Training set: {NUM_TRAIN_PER_CLASS} mèo, {NUM_TRAIN_PER_CLASS} chó (Tổng: {NUM_TRAIN_PER_CLASS*2})")
    print(f"- Test set:     {NUM_TEST_PER_CLASS} mèo, {NUM_TEST_PER_CLASS} chó (Tổng: {NUM_TEST_PER_CLASS*2})")
    print("Đã tạo sẵn ảnh mẫu 'sample_cat.jpg' và 'sample_dog.jpg' để test dự đoán.")
    print("="*50)

if __name__ == "__main__":
    download_file(DATASET_URL, ZIP_PATH)
    extract_and_organize(ZIP_PATH)
