import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('data/processed/colors_clean.csv')
X = df[['red', 'green', 'blue']].values
y = df['primary_label_encoded'].values
class_names = df['primary_label'].unique()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return np.clip(X + noise, 0, 1)

# -----------------------------
# Define models
# -----------------------------
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
}

noise_levels = [0, 0.05, 0.1, 0.2, 0.3]

results = {}
noise_results = {}

# -----------------------------
# Train, evaluate, noise
# -----------------------------
for name, model in models.items():
    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results[name] = {'accuracy': acc, 'f1_macro': f1, 'train_time': train_time}
    
    # Noise experiment
    noise_acc = []
    for nl in noise_levels:
        X_train_noisy = add_noise(X_train, nl)
        X_test_noisy = add_noise(X_test, nl)
        model.fit(X_train_noisy, y_train)
        y_pred_noisy = model.predict(X_test_noisy)
        noise_acc.append(accuracy_score(y_test, y_pred_noisy))
    noise_results[name] = noise_acc

# -----------------------------
# Single figure: Accuracy vs F1 + Noise
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy vs F1
ax = axes[0]
for name in models.keys():
    ax.scatter(results[name]['accuracy'], results[name]['f1_macro'], s=100, label=name)
ax.set_xlabel('Accuracy')
ax.set_ylabel('F1-macro')
ax.set_title('Accuracy vs F1-macro')
ax.grid(True, alpha=0.3)
ax.legend()

# Noise robustness
ax = axes[1]
for name in models.keys():
    ax.plot([nl*100 for nl in noise_levels], noise_results[name], 'o-', label=name)
ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('Accuracy')
ax.set_title('Noise Robustness')
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
