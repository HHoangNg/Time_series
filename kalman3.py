import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# Bước 1: Đọc và xử lý dữ liệu
train_df = pd.read_csv("train.csv")
train_series = train_df['Listening_Time_minutes'].dropna().reset_index(drop=True)

# Bước 2: Khởi tạo Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[train_series.iloc[0]], [0.]])       # initial state: [value, rate]
kf.F = np.array([[1., 1.], [0., 1.]])                 # state transition
kf.H = np.array([[1., 0.]])                           # measurement function
kf.P *= 1000.                                         # initial uncertainty
kf.R = np.var(train_series)                           # measurement noise từ train
kf.Q = np.array([[0.01, 0], [0, 0.01]])               # process noise nhỏ

# Bước 3: Áp dụng Kalman Filter trên train
filtered = []
for z in train_series:
    kf.predict()
    kf.update(z)
    filtered.append(kf.x[0, 0])

# Bước 4: Trực quan hóa
plt.plot(train_series[:500], label='Original (First 500)', color='gray', alpha=0.5)
plt.plot(filtered[:500], label='Kalman Smoothed', color='blue')
plt.title('Kalman Filter on Training Data (Listening Time)')
plt.xlabel('Sample Index')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.show()