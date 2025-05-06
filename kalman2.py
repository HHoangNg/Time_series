from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("train.csv")
series = df['Listening_Time_minutes'].dropna().reset_index(drop=True)
obs = series.values

# Thiết lập mô hình Kalman
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[obs[0]], [0.]])         # initial state: [position, velocity]
kf.F = np.array([[1., 1.], [0., 1.]])     # state transition matrix
kf.H = np.array([[1., 0.]])               # measurement matrix
kf.P *= 1000.                             # covariance matrix

# Đặt nhiễu theo thống kê của chuỗi
kf.R = np.var(obs)                        # measurement noise
kf.Q = np.array([[0.01, 0], [0, 0.01]])   # small process noise for smoothing

# Lọc chuỗi
filtered = []
for z in obs:
    kf.predict()
    kf.update(z)
    filtered.append(kf.x[0, 0])

# Trực quan hóa
plt.plot(obs[:500], label='Original (First 500)', color='gray')
plt.plot(filtered[:500], label='Adaptive Kalman Smoothed', color='blue')
plt.title('Adaptive Kalman Filter on Listening Time (Full Data)')
plt.xlabel('Sample Index')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.show()
