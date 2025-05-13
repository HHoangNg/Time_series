import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import matplotlib.pyplot as plt
import time

# === TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ===
train_file_path = './data/train.csv'
train_data = pd.read_csv(train_file_path)
le = LabelEncoder()
train_data['Activity_encoded'] = le.fit_transform(train_data['Activity'])

X_raw = train_data.drop(columns=["Activity", "subject", "Activity_encoded"]).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
train_data.iloc[:, :-3] = X_scaled

# === CHIA THEO SUBJECT ===
sequences, labels = [], []
for subject in train_data["subject"].unique():
    sub_df = train_data[train_data["subject"] == subject]
    sequences.append(sub_df.drop(columns=["Activity", "subject", "Activity_encoded"]).values)
    labels.append(sub_df["Activity_encoded"].values)

# === THAM SỐ ===
n_states = len(np.unique(train_data["Activity_encoded"]))
n_features = sequences[0].shape[1]
n_particles = 50  # Giảm số particles để tăng tốc độ

# === KHỞI TẠO HMM NGẪU NHIÊN ===
np.random.seed(42)
A = np.random.dirichlet(np.ones(n_states), size=n_states)
means = np.random.randn(n_states, n_features)
covars = np.array([np.eye(n_features) for _ in range(n_states)])

# === HÀM TỐI ƯU HÓA ===
def precompute_logpdf_terms(means, covars):
    inv_covars, log_dets = [], []
    d = means[0].shape[0]
    const = d * np.log(2 * np.pi)
    for cov in covars:
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        inv_covars.append(inv)
        log_dets.append(logdet)
    return inv_covars, log_dets, const

def fast_logpdf(x, mean, inv_cov, log_det, const):
    delta = x - mean
    return -0.5 * (delta @ inv_cov @ delta + log_det + const)

def particle_log_likelihood(seq, A, means, covars, n_particles=50):
    T = len(seq)
    inv_covars, log_dets, const = precompute_logpdf_terms(means, covars)
    ll_list = []

    for _ in range(n_particles):
        s = np.random.choice(n_states)
        log_p = 0
        for t in range(T):
            obs = seq[t]
            log_p += fast_logpdf(obs, means[s], inv_covars[s], log_dets[s], const)
            s = np.random.choice(n_states, p=A[s])
        ll_list.append(log_p)

    return np.mean(ll_list)

# === TRAIN HMM (Particle Filter) ===
start_time = time.time()  # Bắt đầu tính thời gian huấn luyện

best_ll = -np.inf
for _ in range(3):  # Thử 3 lần để giảm thời gian
    A_try = np.random.dirichlet(np.ones(n_states), size=n_states)
    means_try = np.random.randn(n_states, n_features)
    covars_try = np.array([np.eye(n_features) for _ in range(n_states)])

    total_ll = sum(particle_log_likelihood(seq, A_try, means_try, covars_try, n_particles=50) for seq in sequences)

    if total_ll > best_ll:
        A, means, covars = A_try, means_try, covars_try
        best_ll = total_ll

# === DỰ ĐOÁN HIDDEN STATE ===
def predict_states(seq, A, means, covars, n_particles=50):
    T = len(seq)
    particles = np.random.choice(n_states, size=n_particles)
    states = []

    inv_covars, log_dets, const = precompute_logpdf_terms(means, covars)

    for t in range(T):
        obs = seq[t]
        log_weights = np.array([
            fast_logpdf(obs, means[s], inv_covars[s], log_dets[s], const)
            for s in particles
        ])
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        majority_state = mode(particles, keepdims=False).mode
        states.append(majority_state)
        particles = np.array([np.random.choice(n_states, p=A[s]) for s in particles])
    return np.array(states)

# === ÁNH XẠ hidden state → nhãn thật ===
hidden_states_all = [predict_states(seq, A, means, covars, n_particles=50) for seq in sequences]

state_to_label = {}
for pred_seq, true_seq in zip(hidden_states_all, labels):
    for s in np.unique(pred_seq):
        if s not in state_to_label:
            state_to_label[s] = mode(true_seq[pred_seq == s], keepdims=False).mode

# === DỰ ĐOÁN CUỐI ===
y_pred = np.concatenate([[state_to_label.get(s, -1) for s in seq] for seq in hidden_states_all])
y_true = np.concatenate(labels)

# === IN CÁC CHỈ SỐ ===
accuracy = accuracy_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)

# Tính độ ổn định: phần trăm các trạng thái liên tiếp giống nhau (transition ít)
stability = np.mean([
    np.sum(seq[:-1] == seq[1:]) / (len(seq) - 1)
    for seq in hidden_states_all if len(seq) > 1
])

# Tính thời gian huấn luyện
training_time = time.time() - start_time

print(f"Log-likelihood: {best_ll:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Training Time (seconds): {training_time:.2f}")
print(f"Stability: {stability:.4f}")


# === VẼ ===
seq_idx = 0
plt.figure(figsize=(15, 4))
plt.plot(labels[seq_idx], label="True", marker='o')
plt.plot([state_to_label.get(s, -1) for s in hidden_states_all[seq_idx]], label="Predicted", marker='x')
plt.title("Particle Filter HMM")
plt.xlabel("Time step")
plt.ylabel("Activity (encoded)")
plt.legend()
plt.tight_layout()
plt.show()
