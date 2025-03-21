
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# : Đọc dữ liệu
file_path = "d:/AI/Artificial_Neural_Network_Case_Study_data.csv" 
df = pd.read_csv(file_path)

#  Xử lý dữ liệu
# Loại bỏ cột không cần thiết
customer_info = df[['CustomerId', 'Surname']]  
X = df.drop(columns=['Exited', 'CustomerId', 'Surname', 'RowNumber']) 
y = df['Exited']  # Nhãn cần dự đoán

# Mã hóa biến phân loại
X = pd.get_dummies(X, columns=['Gender', 'Geography'], drop_first=True)

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Chia dữ liệu thành tập huấn luyện & kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#  Xây dựng mô hình mạng neuron (ANN)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  
    keras.layers.Dense(8, activation='relu'),  
    keras.layers.Dense(1, activation='sigmoid') 
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# : Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# : Dự đoán trên toàn bộ dữ liệu (10.000 khách hàng)
y_pred_prob_full = model.predict(X_scaled) 
y_pred_binary_full = (y_pred_prob_full > 0.5).astype(int) 

# : Xuất kết quả đầy đ
results_full = customer_info.copy()  
results_full['Thực tế'] = y.values  
results_full['Dự đoán'] = y_pred_binary_full.flatten()  
# Xuất toàn bộ dữ liệu ra file CSV
output_file_full = "d:/AI/KetQuaDuDoan_Full.csv"
results_full.to_csv(output_file_full, index=False)
print(f" Kết quả đã được lưu vào file: {output_file_full}")

# : Xuất 10.000 khách hàng đầu tiên (nếu có đủ )
print(results_full.head(10000))

# : Lưu mô hình mạng neuro
model.save("d:/AI/Artificial_Neural_Network_Case_Study_model.h5")
print(" Mô hình mạng neuron đ đã được lưu vào file: Artificial_Neural_Network_Case_Study_model.h5")

# : Đọc mô hình mạng neuro
model = keras.models.load_model("d:/AI/Artificial_Neural_Network_Case_Study_model.h5")
print(" Mô hình mạng neuron đã đọc từ file: Artificial_Neural_Network_Case_Study_model.h5")

# : Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}")

# : Dự đoán trên toàn bộ dữ liệu 
y_pred_prob_full = model.predict(X_scaled) 
y_pred_binary_full = (y_pred_prob_full > 0.5).astype(int)  

# : Xuất kết quả đày đủ

results_full = customer_info.copy()
results_full['Thực tế'] = y.values
results_full['Dự đoán'] = y_pred_binary_full.flatten()

# Xuất toàn bộ dữ liệu ra file CSV
output_file_full = "d:/AI/KetQuaDuDoan_Full.csv"
results_full.to_csv(output_file_full, index=False)
print(f" Kết quả được lưu vào file: {output_file_full}")
