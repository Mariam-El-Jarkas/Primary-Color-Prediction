import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_splits():
    splits_path = 'data/processed/splits/splits.pkl'
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    return splits

def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level * 255, X.shape)
    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0, 255)
    return X_noisy

def train_decision_tree_with_noise(max_depth_values=[3, 5, 7, 10, 15, None], noise_levels=[0, 0.05, 0.1, 0.2]):
    print("🚀 Training Decision Tree Model with Noise Experiment")
    print("="*60)
    
    splits = load_splits()
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_test = splits['y_test']
    label_encoder = splits['label_encoder']
    class_names = splits['class_names']
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Classes: {len(class_names)}")
    print(f"Using class_weight='balanced' to handle class imbalance")
    
    results = {}
    noise_results = {}
    
    for max_depth in max_depth_values:
        depth_str = str(max_depth) if max_depth is not None else "None"
        print(f"\n🔧 Training Decision Tree with max_depth={depth_str} (no noise)...")
        
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            criterion='gini',
            class_weight='balanced'
        )
        dt_model.fit(X_train, y_train)
        
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, dt_model.predict(X_train))
        
        results[max_depth] = {
            'model': dt_model,
            'accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'predictions': y_pred,
            'true_labels': y_test,
            'class_names': class_names
        }
        
        print(f"   Test Accuracy: {accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")
    
    best_depth = max(results.keys(), key=lambda d: results[d]['accuracy'])
    print(f"\n🔬 Noise Experiment with best depth={best_depth}:")
    
    for noise_level in noise_levels:
        if noise_level == 0:
            continue
            
        print(f"\n📊 Noise Level: {noise_level*100:.0f}%")
        X_train_noisy = add_noise(X_train, noise_level)
        X_test_noisy = add_noise(X_test, noise_level)
        
        dt_noisy = DecisionTreeClassifier(
            max_depth=best_depth,
            random_state=42,
            criterion='gini',
            class_weight='balanced'
        )
        dt_noisy.fit(X_train_noisy, y_train)
        
        y_pred_noisy = dt_noisy.predict(X_test_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        
        noise_results[noise_level] = {
            'model': dt_noisy,
            'accuracy': accuracy_noisy,
            'predictions': y_pred_noisy,
            'true_labels': y_test
        }
        
        baseline_acc = results[best_depth]['accuracy']
        accuracy_change = accuracy_noisy - baseline_acc
        print(f"   Accuracy: {accuracy_noisy:.4f} (Δ = {accuracy_change:+.4f})")
    
    plot_decision_tree_results(results, noise_results, best_depth)
    
    print(f"\n🔍 Feature Importance (no noise):")
    best_model = results[best_depth]['model']
    for name, importance in zip(['Red', 'Green', 'Blue'], best_model.feature_importances_):
        print(f"   {name}: {importance:.4f}")
    
    print(f"\n📊 Class-wise Performance (Decision Tree with class_weight='balanced'):")
    y_pred_best = results[best_depth]['predictions']
    print(classification_report(y_test, y_pred_best, target_names=class_names))
    
    return results, best_depth, noise_results

def plot_decision_tree_results(results, noise_results, best_depth):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    depths = list(results.keys())
    depth_labels = [str(d) if d is not None else "None" for d in depths]
    test_accuracies = [results[d]['accuracy'] for d in depths]
    train_accuracies = [results[d]['train_accuracy'] for d in depths]
    
    axes[0].plot(depth_labels, train_accuracies, 'bo-', linewidth=2, markersize=8, label='Train')
    axes[0].plot(depth_labels, test_accuracies, 'ro-', linewidth=2, markersize=8, label='Test')
    axes[0].set_xlabel('Tree Depth', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Decision Tree: Accuracy vs Depth (No Noise)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    best_idx = depths.index(best_depth)
    best_acc = results[best_depth]['accuracy']
    axes[0].plot(depth_labels[best_idx], best_acc, 'g*', markersize=15, 
                label=f'Best: depth={best_depth}, acc={best_acc:.3f}')
    axes[0].legend()
    
    if noise_results:
        noise_levels = list(noise_results.keys())
        noise_accuracies = [noise_results[nl]['accuracy'] for nl in noise_levels]
        
        axes[1].plot(noise_levels, noise_accuracies, 'mo-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Noise Level', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'Decision Tree: Accuracy vs Noise (depth={best_depth})', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        baseline_acc = results[best_depth]['accuracy']
        axes[1].axhline(y=baseline_acc, color='blue', linestyle='--', 
                       label=f'Baseline (0 noise): {baseline_acc:.3f}')
        axes[1].legend()
        
        for i, (nl, acc) in enumerate(zip(noise_levels, noise_accuracies)):
            axes[1].annotate(f'{acc:.3f}', (nl, acc), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9)
    
    plt.suptitle('Decision Tree Model Analysis (Handling Class Imbalance)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results, best_depth, noise_results = train_decision_tree_with_noise()
    print("\n✅ Decision Tree model complete with noise experiment!")