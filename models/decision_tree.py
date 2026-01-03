import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv('data/processed/colors_clean.csv')
X = df[['red', 'green', 'blue']].values
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
# Tune max_depth
# -----------------------------
max_depth_values = [3, 5, 7, 10, 15, None]  # None = no limit
results = {}

print("\n🚀 Tuning Decision Tree max_depth...")
print("="*60)

for max_depth in max_depth_values:
    depth_str = str(max_depth) if max_depth is not None else "None"
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        criterion='gini',
        class_weight='balanced'
    )
    dt.fit(X_train, y_train)
    
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    results[max_depth] = {
        'model': dt,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_macro': test_f1,
        'predictions': y_pred_test
    }
    
    print(f"\nMax Depth: {depth_str}")
    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print(f"  F1-macro:  {test_f1:.4f}")

# -----------------------------
# Best max_depth
# -----------------------------
best_depth = max(results.keys(), key=lambda d: results[d]['test_accuracy'])
best_model = results[best_depth]['model']
best_acc = results[best_depth]['test_accuracy']
best_f1 = results[best_depth]['f1_macro']
depth_str = str(best_depth) if best_depth is not None else "None"

print("\n🏆 Best max_depth:", depth_str)
print(f"   Test Accuracy: {best_acc:.4f}")
print(f"   F1-macro: {best_f1:.4f}")

# -----------------------------
# Noise Experiment
# -----------------------------
noise_levels = [0, 0.05, 0.1, 0.2, 0.3]
noise_results = {}

print("\n" + "="*60)
print("🔬 Noise Experiment with Best Model")
print("="*60)

for nl in noise_levels:
    X_train_noisy = add_noise(X_train, nl)
    X_test_noisy = add_noise(X_test, nl)
    
    dt_noisy = DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=42,
        criterion='gini',
        class_weight='balanced'
    )
    dt_noisy.fit(X_train_noisy, y_train)
    y_pred_noisy = dt_noisy.predict(X_test_noisy)
    
    acc_noisy = accuracy_score(y_test, y_pred_noisy)
    f1_noisy = f1_score(y_test, y_pred_noisy, average='macro')
    
    noise_results[nl] = {
        'accuracy': acc_noisy,
        'f1_macro': f1_noisy,
        'predictions': y_pred_noisy
    }
    
    if nl == 0:
        print(f"\n📌 Baseline (No Noise): Acc={acc_noisy:.4f}, F1-macro={f1_noisy:.4f}")
    else:
        delta = acc_noisy - best_acc
        print(f"Noise {nl*100:.0f}%: Acc={acc_noisy:.4f} (Δ={delta:+.4f}), F1-macro={f1_noisy:.4f}")

# -----------------------------
# Plots
# -----------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Train/Test Accuracy vs max_depth
depth_labels = [str(d) if d is not None else "None" for d in max_depth_values]
train_accs = [results[d]['train_accuracy'] for d in max_depth_values]
test_accs = [results[d]['test_accuracy'] for d in max_depth_values]

ax1.plot(depth_labels, train_accs, 'bo-', label='Train Acc', linewidth=2)
ax1.plot(depth_labels, test_accs, 'ro-', label='Test Acc', linewidth=2)
ax1.plot(depth_labels[depth_labels.index(depth_str)], best_acc, 'g*', markersize=15, label='Best Model')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Accuracy')
ax1.set_title('Decision Tree: Accuracy vs max_depth')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Noise Impact
ax2.plot([nl*100 for nl in noise_levels], [noise_results[nl]['accuracy'] for nl in noise_levels], 'mo-', label='Accuracy')
ax2.plot([nl*100 for nl in noise_levels], [noise_results[nl]['f1_macro'] for nl in noise_levels], 'co-', label='F1-macro')
ax2.axhline(y=best_acc, color='gray', linestyle='--', label=f'Baseline: {best_acc:.3f}')
ax2.set_xlabel('Noise Level (%)')
ax2.set_ylabel('Metric Value')
ax2.set_title(f'Noise Impact (max_depth={depth_str})')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Feature Importance
feature_names = ['Red', 'Green', 'Blue']
importances = best_model.feature_importances_
ax3.bar(feature_names, importances, color=['red','green','blue'], alpha=0.7)
ax3.set_xlabel('Feature')
ax3.set_ylabel('Importance')
ax3.set_title('Feature Importance (Gini)')
ax3.grid(True, axis='y', alpha=0.3)

# Confusion Matrix
y_pred_best = results[best_depth]['predictions']
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title('Confusion Matrix')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('True')

plt.suptitle('Decision Tree Model Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------
# Tree Visualization (small depth)
# -----------------------------
if best_depth is not None and best_depth <= 5:
    plt.figure(figsize=(10,6))
    plot_tree(best_model, feature_names=feature_names, class_names=[str(c) for c in class_names],
              filled=True, rounded=True, max_depth=3)
    plt.title(f'Decision Tree Visualization (max_depth={depth_str})')
    plt.show()
else:
    print(f"⚠️ Tree too large to visualize (depth={depth_str})")

# -----------------------------
# Classification Report
# -----------------------------
print("\n" + "="*60)
print("📊 Classification Report (Best Model)")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=class_names))


