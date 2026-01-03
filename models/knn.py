import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load and clean data
# -----------------------------
df = pd.read_csv('data/processed/colors_clean.csv')

# Remove extreme dark/light colors for KNN
print("Removing extreme colors for KNN...")
df = df[
    (df['red'] >= 0.05) & (df['red'] <= 0.95) &
    (df['green'] >= 0.05) & (df['green'] <= 0.95) &
    (df['blue'] >= 0.05) & (df['blue'] <= 0.95)
].copy()

print(f"Using {len(df)} colors (removed extremes)")

X = df[['red', 'green', 'blue']].values
y = df['primary_label_encoded'].values
class_names = df['primary_label'].unique()

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Classes: {len(class_names)}")

# -----------------------------
# Add noise function
# -----------------------------
def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0, 1)
    return X_noisy

# -----------------------------
# Train KNN
# -----------------------------
k_values = [3, 5, 7, 9, 11]
noise_levels = [0, 0.05, 0.1, 0.2]
results = {}
noise_results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='macro')
    
    if train_acc - test_acc > 0.05:
        fit_status = "Overfit"
    elif test_acc < 0.7:
        fit_status = "Underfit"
    else:
        fit_status = "Good Fit"
    
    results[k] = {
        'train_accuracy': train_acc,
        'accuracy': test_acc,
        'f1_score': f1,
        'fit_status': fit_status
    }
    
    print(f"k={k}: Train={train_acc:.4f}, Test={test_acc:.4f}, F1={f1:.4f}, Fit={fit_status}")

# Best k
best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest k: {best_k} → Test={results[best_k]['accuracy']:.4f}")

# -----------------------------
# Noise test
# -----------------------------
for noise_level in noise_levels:
    X_train_noisy = add_noise(X_train, noise_level)
    X_test_noisy = add_noise(X_test, noise_level)
    
    knn_noisy = KNeighborsClassifier(n_neighbors=best_k)
    knn_noisy.fit(X_train_noisy, y_train)
    y_pred_noisy = knn_noisy.predict(X_test_noisy)
    
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
    f1_noisy = f1_score(y_test, y_pred_noisy, average='macro')
    
    noise_results[noise_level] = {
        'accuracy': accuracy_noisy,
        'f1_score': f1_noisy
    }
    
    print(f"Noise {noise_level*100:.0f}% → Acc={accuracy_noisy:.4f}")

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(list(results.keys()), [results[k]['accuracy'] for k in results], 'bo-', label='Test')
plt.plot(list(results.keys()), [results[k]['train_accuracy'] for k in results], 'go--', label='Train')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs k')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot([nl*100 for nl in noise_levels], [noise_results[nl]['accuracy'] for nl in noise_levels], 'ro-')
plt.xlabel('Noise Level (%)')
plt.ylabel('Accuracy')
plt.title(f'Noise Impact (k={best_k})')
plt.grid(True)
plt.show()

# -----------------------------
# Confusion matrix
# -----------------------------
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (k={best_k})')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))