import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from einops import rearrange

# ====== 1. Load và tiền xử lý dữ liệu =======
df = pd.read_csv("D:/python/Autoformer and Gaussian/data/weather_data.csv")

# Kiểm tra dữ liệu
print("Columns:", df.columns)

# Loại bỏ cột 'location' nếu có
if 'location' in df.columns:
    df = df.drop(columns=['location'])

# Chuyển cột thời gian thành datetime nếu có
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.drop(columns=['date'])

# Lọc các cột có kiểu số
df = df.select_dtypes(include=[np.number])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)

# Thêm Gaussian noise
def add_gaussian_noise(data, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

noisy_data = add_gaussian_noise(scaled_data)

# ====== 2. Dataset & Dataloader =======
class TempDataset(Dataset):
    def __init__(self, data, input_len, pred_len):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.pred_len, 0]  # Dự đoán nhiệt độ (giả sử cột 0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

input_len = 36
pred_len = 9

train_size = int(len(noisy_data) * 0.7)
val_size = int(len(noisy_data) * 0.2)

train_data = noisy_data[:train_size]
val_data = noisy_data[train_size:train_size+val_size]
test_data = noisy_data[train_size+val_size:]

train_dataset = TempDataset(train_data, input_len, pred_len)
val_dataset = TempDataset(val_data, input_len, pred_len)
test_dataset = TempDataset(test_data, input_len, pred_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ====== 3. Định nghĩa Autoformer =======
class AutoCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Đây là simplified correlation: bạn có thể thay thế bằng hàm Attention nếu cần
        return x

class AutoformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.correlation = AutoCorrelation()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.correlation(x)
        x = x + x2
        x = self.norm1(x)

        x2 = self.ffn(x)
        x = x + x2
        x = self.norm2(x)
        return x

class Autoformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_blocks=2, pred_len=12):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder = nn.Sequential(*[AutoformerBlock(d_model) for _ in range(num_blocks)])
        self.decoder = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, x):
        # x shape: (B, T, input_dim)
        x = self.embedding(x)  # -> (B, T, d_model)
        x = self.encoder(x)
        x = self.decoder(x)  # -> (B, T, 1)
        out = x[:, -self.pred_len:, 0]  # lấy pred_len bước cuối
        return out

# ====== 4. Huấn luyện =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoformer(input_dim=scaled_data.shape[1], pred_len=pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(2):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

# ====== 5. Đánh giá trên tập test =======
model.eval()
preds = []
trues = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        preds.append(out.flatten())
        trues.append(y.numpy().flatten())

preds = np.array(preds).flatten()
trues = np.array(trues).flatten()

rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)

print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# ====== 6. Vẽ biểu đồ kết quả =======
plt.figure(figsize=(12,4))
plt.plot(trues[:200], label="True")
plt.plot(preds[:200], label="Predicted")
plt.legend()
plt.title("Temperature Prediction")
plt.show()
