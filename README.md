# Phân Loại X-ray COVID 🦠

Dự án này là một ứng dụng web sử dụng học sâu (Deep Learning) để phân loại hình ảnh X-ray phổi thành các loại bệnh khác nhau, bao gồm **COVID-19**, **Bacterial**, **Lung Opacity**, **Normal**, và **Viral**. Ứng dụng được xây dựng bằng **Flask** và tích hợp mô hình học sâu được huấn luyện bằng **TensorFlow/Keras**.

---

## 🚀 Tính Năng

- **Tải lên ảnh X-ray**: Người dùng có thể tải lên ảnh X-ray phổi từ máy tính.
- **Dự đoán bệnh**: Ứng dụng sẽ phân loại ảnh X-ray và trả về xác suất dự đoán cho từng loại bệnh.
- **Hiển thị kết quả trực quan**: Kết quả dự đoán được hiển thị với giao diện đẹp mắt.
- **Mô hình học sâu**: Sử dụng mô hình CNN (Convolutional Neural Network) được huấn luyện trên tập dữ liệu X-ray.

---

## 🛠️ Công Nghệ Sử Dụng

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Học sâu**: TensorFlow/Keras
- **Thư viện khác**: NumPy, PIL, Matplotlib, Seaborn, Scikit-learn

---

## 📂 Cấu Trúc Dự Án

```
Github_phanloaibenhphoi/
│
├── flaskProject_model_X-ray/
│   ├── app.py                # File chính chạy Flask app
│   ├── static/
│   │   ├── styles.css        # File CSS cho giao diện
│   │   ├── script.js         # File JavaScript xử lý logic frontend
│   │   └── uploads/          # Thư mục lưu ảnh tải lên
│   ├── templates/
│   │   └── index.html        # Giao diện chính của ứng dụng
│   └── .gitignore            # File cấu hình Git
│
├── Create_Model/
│   ├── create_model_456.py   # File huấn luyện mô hình
│   ├── Test_model.py         # File kiểm tra mô hình
│   ├── Data.txt              # Link tập dữ liệu
│   └── Bao_Cao_CNN.pptx      # Báo cáo mô hình
│
└── .idea/                    # Cấu hình IDE (PyCharm)
```

---

## 📦 Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/Github_phanloaibenhphoi.git
cd Github_phanloaibenhphoi
```

### 2. Tạo Môi Trường Ảo
```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/MacOS
venv\Scripts\activate     # Trên Windows
```

### 3. Cài Đặt Thư Viện
```bash
pip install -r requirements.txt
```

### 4. Chuẩn Bị Mô Hình
- Huấn luyện mô hình bằng file `create_model_456.py` hoặc tải mô hình đã huấn luyện sẵn.
- Đặt file mô hình (`X-ray-covid_model_3_v2.h5`) vào thư mục `flaskProject_model_X-ray`.

### 5. Chạy Ứng Dụng
```bash
cd flaskProject_model_X-ray
python app.py
```
Ứng dụng sẽ chạy tại `http://127.0.0.1:5000`.

---

## 🖼️ Giao Diện Ứng Dụng

### Trang Chính
![Trang Chính](https://via.placeholder.com/800x400?text=Trang+Ch%C3%ADnh)

### Kết Quả Dự Đoán
![Kết Quả Dự Đoán](https://via.placeholder.com/800x400?text=K%E1%BA%BFt+Qu%E1%BA%A3+D%E1%BB%B1+%C4%91o%C3%A1n)

---

## 📊 Huấn Luyện Mô Hình

### Tập Dữ Liệu
- Tập dữ liệu được sử dụng: [COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases).

### Kiến Trúc Mô Hình
- Mô hình CNN với các lớp:
  - **Convolutional Layers**: 3 lớp
  - **Pooling Layers**: 3 lớp
  - **Dense Layers**: 2 lớp
  - **Dropout**: 0.4 để giảm overfitting

### Kết Quả Huấn Luyện
- **Độ chính xác trên tập kiểm tra**: ~90%
- **Mất mát trên tập kiểm tra**: ~0.3

---

## 🧪 Kiểm Tra Mô Hình

- **Confusion Matrix**:
  ![Confusion Matrix](https://via.placeholder.com/800x400?text=Confusion+Matrix)

- **Precision-Recall Curve**:
  ![Precision-Recall Curve](https://via.placeholder.com/800x400?text=Precision-Recall+Curve)

---

## 📝 Ghi Chú

- Đảm bảo rằng bạn đã cài đặt **TensorFlow** và **CUDA** nếu sử dụng GPU.
- Tập dữ liệu cần được đặt đúng đường dẫn trong file `create_model_456.py`.

---

## 📄 Giấy Phép

Dự án này được phát hành dưới giấy phép **MIT License**. Xem chi tiết tại [LICENSE](LICENSE).

---

## 📧 Liên Hệ

Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ qua email: `nguyenhoangdien1x@gmail.com`.
