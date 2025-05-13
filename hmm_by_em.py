import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from hmmlearn import hmm
from scipy.stats import mode

# Đường dẫn đến file
train_file_path = './data/train.csv'
test_file_path = './data/test.csv'

# Đọc dữ liệu
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Mã hóa nhãn hoạt động
label_encoder = LabelEncoder()
train_data['Activity_encoded'] = label_encoder.fit_transform(train_data['Activity'])

# Tách đặc trưng và chuẩn hóa
X = train_data.drop(columns=["Activity", "subject", "Activity_encoded"]).values
y = train_data["Activity_encoded"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tạo chuỗi dữ liệu theo từng subject
sequences, lengths, labels = [], [], []
for subject_id in train_data["subject"].unique():
    subject_data = train_data[train_data["subject"] == subject_id]
    seq = scaler.transform(subject_data.drop(columns=["Activity", "subject", "Activity_encoded"]).values)
    sequences.append(seq)
    lengths.append(len(seq))
    labels.append(subject_data["Activity_encoded"].values)

# Nối các chuỗi thành 1 mảng lớn cho HMM
X_concat = np.concatenate(sequences)
lengths_concat = [len(seq) for seq in sequences]

# Huấn luyện mô hình HMM
n_states = len(np.unique(y))
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)

start_time = time.time()
model.fit(X_concat, lengths=lengths_concat)
training_time = time.time() - start_time

# Dự đoán trạng thái ẩn
hidden_states = model.predict(X_concat, lengths=lengths_concat)

# Ánh xạ từ hidden state -> activity (dựa vào mode)
state_to_activity = {}
start = 0
for i, length in enumerate(lengths_concat):
    state_seq = hidden_states[start:start+length]
    true_seq = labels[i]
    for s in np.unique(state_seq):
        most_common_activity = mode(true_seq[state_seq == s], keepdims=False).mode
        if s not in state_to_activity:
            state_to_activity[s] = most_common_activity
    start += length

# Gán nhãn dự đoán theo hidden state
y_pred = np.array([state_to_activity.get(s, -1) for s in hidden_states])
y_true = np.concatenate(labels)

# Tính các chỉ số
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
log_likelihood = model.score(X_concat, lengths=lengths_concat)
stability = np.mean(hidden_states[1:] == hidden_states[:-1])

# In kết quả
print(f"Log-likelihood: {log_likelihood:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Training Time (seconds): {training_time:.4f}")
print(f"Stability (proportion of unchanged hidden states): {stability:.4f}")

# Trực quan hóa kết quả cho subject đầu tiên
seq_idx = 0
start = sum(lengths[:seq_idx])
end = start + lengths[seq_idx]
time_steps = np.arange(end - start)
true_label = labels[seq_idx]
pred_label = [state_to_activity.get(s, -1) for s in hidden_states[start:end]]

plt.figure(figsize=(15, 4))
plt.plot(time_steps, true_label, label='True Label', marker='o')
plt.plot(time_steps, pred_label, label='Predicted Label (HMM)', marker='x')
plt.title(f"HMM Prediction vs True Label for Subject {train_data['subject'].unique()[seq_idx]}")
plt.xlabel('Time Step')
plt.ylabel('Activity (Encoded)')
plt.legend()
plt.tight_layout()
plt.show()
