
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv('data/processed/colors_clean.csv')
X = df[['red', 'green', 'blue']].values
y = df['primary_label_encoded'].values
class_names = df['primary_label'].unique()

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(class_names)}")

# -----------------------------
# Noise function
# -----------------------------
def add_noise(X, noise_level=0.1, seed=42):
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level, X.shape)
    return np.clip(X + noise, 0, 1)

# -----------------------------
# Hyperparameter tuning
# -----------------------------
n_estimators_values = [50, 100, 150]
max_depth_values = [5, 10, 15, None]  # None = no limit
results = {}

print("\n" + "="*60)
print("🚀 HYPERPARAMETER TUNING")
print("="*60)

for n_est in n_estimators_values:
    for depth in max_depth_values:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[(n_est, depth)] = {
            'model': rf, 
            'accuracy': acc, 
            'f1': f1,
            'importances': rf.feature_importances_
        }
        
        depth_str = str(depth) if depth is not None else "None"
        print(f"n_estimators={n_est:3}, max_depth={depth_str:4} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Select best combination
best_params = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_params]['model']
best_acc = results[best_params]['accuracy']
best_f1 = results[best_params]['f1']

print(f"\n🏆 BEST MODEL: n_estimators={best_params[0]}, max_depth={best_params[1]}")
print(f"   Accuracy: {best_acc:.4f}")
print(f"   F1-macro: {best_f1:.4f}")

# -----------------------------
# Noise Experiment
# -----------------------------
noise_levels = [0, 0.05, 0.1, 0.2, 0.3]
noise_results = {}

print("\n" + "="*60)
print("🔬 NOISE EXPERIMENT")
print("="*60)

for nl in noise_levels:
    X_train_noisy = add_noise(X_train, nl)
    X_test_noisy = add_noise(X_test, nl)
    
    rf_noisy = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_noisy.fit(X_train_noisy, y_train)
    y_pred_noisy = rf_noisy.predict(X_test_noisy)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)
    f1_noisy = f1_score(y_test, y_pred_noisy, average='macro')
    
    noise_results[nl] = {
        'accuracy': acc_noisy, 
        'f1': f1_noisy, 
        'predictions': y_pred_noisy,
        'importances': rf_noisy.feature_importances_
    }
    
    if nl == 0:
        print(f"\n📌 Baseline (No Noise):")
        print(f"  Accuracy: {acc_noisy:.4f}")
        print(f"  F1-macro: {f1_noisy:.4f}")
    else:
        delta = acc_noisy - best_acc
        print(f"Noise {nl*100:.0f}%:")
        print(f"  Accuracy: {acc_noisy:.4f} (Δ = {delta:+.4f})")
        print(f"  F1-macro: {f1_noisy:.4f}")

# -----------------------------
# Feature Importance Analysis
# -----------------------------
feature_names = ['Red', 'Green', 'Blue']
importances = best_model.feature_importances_

print("\n" + "="*60)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*60)
print("\nFeature Importances (No Noise):")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# -----------------------------
# Visualizations (2x2 Grid)
# -----------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy vs Noise
ax1.plot([nl*100 for nl in noise_levels], 
         [noise_results[nl]['accuracy'] for nl in noise_levels], 
         'bo-', linewidth=2, markersize=8, label='Accuracy')
ax1.plot([nl*100 for nl in noise_levels], 
         [noise_results[nl]['f1'] for nl in noise_levels], 
         'ro-', linewidth=2, markersize=8, label='F1-macro')
ax1.axhline(y=best_acc, color='gray', linestyle='--', 
           label=f'Baseline: {best_acc:.3f}')
ax1.set_xlabel('Noise Level (%)')
ax1.set_ylabel('Metric Value')
ax1.set_title(f'Random Forest: Noise Impact\n(n_estimators={best_params[0]}, max_depth={best_params[1]})')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Feature Importance (No Noise)
bars = ax2.bar(feature_names, importances, color=['red', 'green', 'blue'], alpha=0.7)
ax2.set_xlabel('Feature')
ax2.set_ylabel('Importance Score')
ax2.set_title('Feature Importance (No Noise)')
ax2.grid(True, alpha=0.3, axis='y')
for bar, imp in zip(bars, importances):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{imp:.3f}', ha='center', va='bottom', fontsize=10)

# Plot 3: Feature Importance vs Noise
selected_noise = [0, 0.1, 0.3]
importance_data = []
for nl in selected_noise:
    importance_data.append(noise_results[nl]['importances'])

x = np.arange(len(feature_names))
width = 0.25
for i, nl in enumerate(selected_noise):
    offset = (i - 1) * width
    ax3.bar(x + offset, importance_data[i], width, 
            label=f'{nl*100:.0f}% noise' if nl > 0 else '0% noise',
            alpha=0.7)

ax3.set_xlabel('Feature')
ax3.set_ylabel('Importance Score')
ax3.set_title('Feature Importance vs Noise Level')
ax3.set_xticks(x)
ax3.set_xticklabels(feature_names)
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()

# Plot 4: Confusion Matrix (No Noise)
y_pred_best = noise_results[0]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
ax4.set_title('Confusion Matrix (No Noise)')
ax4.set_xlabel('Predicted Label')
ax4.set_ylabel('True Label')
ax4.set_xticks(range(len(class_names)))
ax4.set_yticks(range(len(class_names)))
ax4.set_xticklabels(class_names, rotation=45, ha='right')
ax4.set_yticklabels(class_names)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.suptitle('Random Forest Model Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------
# Classification Report
# -----------------------------
print("\n" + "="*60)
print("📊 CLASSIFICATION REPORT (No Noise)")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=class_names))

print("\n✅ Random Forest experiment complete!")
