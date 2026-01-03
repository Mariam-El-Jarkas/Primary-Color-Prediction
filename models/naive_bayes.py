import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv('data/processed/colors_clean.csv')
X = df[['red','green','blue']].values
y = df['primary_label_encoded'].values
class_names = df['primary_label'].unique()

# -----------------------------
# Stratified train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(class_names)}")

# -----------------------------
# Noise function
# -----------------------------
def add_noise(X, noise_level=0.1):
    """Add Gaussian noise to features, clipped to 0-1."""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)

# -----------------------------
# Fit evaluation
# -----------------------------
def evaluate_fit(train_acc, test_acc):
    """Determine if model is overfit, underfit, or best fit."""
    if train_acc - test_acc > 0.05:
        return "Overfit"
    elif test_acc < 0.7:
        return "Underfit"
    else:
        return "Best Fit"

# -----------------------------
# Train Naive Bayes & Noise Experiment
# -----------------------------
noise_levels = [0, 0.05, 0.1, 0.2, 0.3]
results = {}

print("\n🚀 Training Naive Bayes Model with Noise Experiment")
print("="*60)

# Train baseline first
nb_baseline = GaussianNB()
nb_baseline.fit(X_train, y_train)
y_pred_train = nb_baseline.predict(X_train)
train_acc_baseline = accuracy_score(y_train, y_pred_train)
y_pred_test = nb_baseline.predict(X_test)
test_acc_baseline = accuracy_score(y_test, y_pred_test)
fit_status = evaluate_fit(train_acc_baseline, test_acc_baseline)

results[0] = {
    'model': nb_baseline,
    'accuracy': test_acc_baseline,
    'f1_macro': f1_score(y_test, y_pred_test, average='macro'),
    'predictions': y_pred_test,
    'train_accuracy': train_acc_baseline,
    'fit': fit_status
}

print(f"\n📌 Baseline (No Noise) → Train Acc: {train_acc_baseline:.4f}, Test Acc: {test_acc_baseline:.4f}, F1-macro: {results[0]['f1_macro']:.4f}, Fit: {fit_status}")

# Noise experiments
for nl in noise_levels[1:]:
    X_train_noisy = add_noise(X_train, nl)
    X_test_noisy = add_noise(X_test, nl)
    
    nb = GaussianNB()
    nb.fit(X_train_noisy, y_train)
    y_pred_train_noisy = nb.predict(X_train_noisy)
    y_pred_noisy = nb.predict(X_test_noisy)
    
    train_acc = accuracy_score(y_train, y_pred_train_noisy)
    test_acc = accuracy_score(y_test, y_pred_noisy)
    f1_macro = f1_score(y_test, y_pred_noisy, average='macro')
    fit_status = evaluate_fit(train_acc, test_acc)
    
    results[nl] = {
        'model': nb,
        'accuracy': test_acc,
        'f1_macro': f1_macro,
        'predictions': y_pred_noisy,
        'train_accuracy': train_acc,
        'fit': fit_status
    }
    
    print(f"Noise {nl*100:.0f}% → Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, F1-macro: {f1_macro:.4f}, Fit: {fit_status}")

# -----------------------------
# Summary Table
# -----------------------------
print("\n📊 Summary of Noise Experiment:")
print("Noise Level (%) | Train Acc | Test Acc | F1-macro | Fit Status")
print("---------------------------------------------------------------")
for nl in noise_levels:
    r = results[nl]
    print(f"{nl*100:>13.0f} | {r['train_accuracy']:.4f}   | {r['accuracy']:.4f}   | {r['f1_macro']:.4f} | {r['fit']}")

# -----------------------------
# Confusion matrix for baseline
# -----------------------------
y_pred_clean = results[0]['predictions']
cm = confusion_matrix(y_test, y_pred_clean)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Naive Bayes Confusion Matrix (No Noise)')
plt.show()

# -----------------------------
# Plot Noise Impact
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot([nl*100 for nl in noise_levels], [results[nl]['accuracy'] for nl in noise_levels], 'go-', label='Accuracy')
plt.plot([nl*100 for nl in noise_levels], [results[nl]['f1_macro'] for nl in noise_levels], 'ro-', label='F1-macro')
plt.xlabel('Noise Level (%)')
plt.ylabel('Metric')
plt.title('Naive Bayes: Accuracy & F1 vs Noise')
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Classification report (baseline)
# -----------------------------
print("\nClassification Report (No Noise):")
print(classification_report(y_test, y_pred_clean, target_names=class_names))
