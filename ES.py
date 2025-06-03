import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Bước 1: Đọc dữ liệu
file_path = "DOM_hourly.csv"
df = pd.read_csv(file_path, parse_dates=['Datetime'])
df = df.sort_values('Datetime')
df.set_index('Datetime', inplace=True)
series = df['DOM_MW']

# --- Bước 2: Lấy đoạn dữ liệu gần nhất để huấn luyện nhanh
series = series[-5000:]  # bạn có thể tăng nếu máy mạnh

# --- Bước 3: Khởi tạo mô hình Holt-Winters (Exponential Smoothing)
# Giả sử dữ liệu có chu kỳ theo ngày (24 giờ)
model = ExponentialSmoothing(
    series,
    trend='add',
    seasonal='add',
    seasonal_periods=24
)

# --- Bước 4: Huấn luyện mô hình
fitted_model = model.fit()

# --- Bước 5: Dự báo 100 bước tiếp theo
forecast = fitted_model.forecast(100)

# --- Bước 6: Trực quan hóa kết quả
plt.figure(figsize=(14, 5))
plt.plot(series, label='Original')
plt.plot(forecast, label='Forecast (Exponential Smoothing)', color='orange')
plt.title("Dự báo công suất tiêu thụ điện - Exponential Smoothing")
plt.xlabel("Datetime")
plt.ylabel("DOM_MW (MW)")
plt.legend()
plt.show()
