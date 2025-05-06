import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
# Đọc dữ liệu
df = pd.read_csv("train.csv")

# Lấy chuỗi Listening_Time_minutes, loại bỏ giá trị thiếu
listening_series = df['Listening_Time_minutes'].dropna().reset_index(drop=True)


observations = listening_series.values

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[observations[0]], [0.]])   # initial state
kf.F = np.array([[1., 1.], [0., 1.]])         # state transition
kf.H = np.array([[1., 0.]])                  # measurement function
kf.P *= 1000.                                # initial uncertainty
kf.R = 5                                     # measurement noise
kf.Q = np.array([[1, 0], [0, 1]])            # process noise

filtered = []
for z in observations:
    kf.predict()
    kf.update(z)
    filtered.append(kf.x[0, 0])

plt.plot(observations[:500], label='Original (First 500)', color='gray')
plt.plot(filtered[:500], label='Kalman Smoothed', color='blue')
plt.title('Kalman Filter on Listening Time (Full Data)')
plt.xlabel('Sample Index')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.show()
