

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
    X_noisy = np.clip(X_noisy, 0, 255)  # Keep in valid RGB range
    return X_noisy

def train_knn_model_with_noise(k_values=[3, 5, 7, 9, 11], noise_levels=[0, 0.05, 0.1, 0.2]):

    print("🚀 Training K-Nearest Neighbors Model with Noise Experiment")
    print("="*60)
    
    # Load data
    splits = load_splits()
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_test = splits['y_test']
    label_encoder = splits['label_encoder']
    class_names = splits['class_names']
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(class_names)}")
    
    results = {}
    noise_results = {}
    
    # Train without noise first
    for k in k_values:
        print(f"\n🔧 Training KNN with k={k} (no noise)...")
        
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[k] = {
            'model': knn,
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Noise experiment
    print(f"\n🔬 Noise Experiment (varying noise levels):")
    for noise_level in noise_levels:
        print(f"\n📊 Noise Level: {noise_level*100:.0f}%")
        
        X_train_noisy = add_noise(X_train, noise_level)
        X_test_noisy = add_noise(X_test, noise_level)
        
        for k in [5]:  # Test with k=5 for noise experiment
            knn_noisy = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn_noisy.fit(X_train_noisy, y_train)
            
            y_pred_noisy = knn_noisy.predict(X_test_noisy)
            accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
            
            noise_results[noise_level] = {
                'accuracy': accuracy_noisy,
                'k': k
            }
            
            print(f"   k={k}: Accuracy = {accuracy_noisy:.4f} "
                  f"(Δ = {accuracy_noisy - results[5]['accuracy']:+.4f})")
    
    # Find best k (without noise)
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_k]['accuracy']
    
    print(f"\n🏆 Best Model (no noise): k={best_k} with accuracy={best_accuracy:.4f}")
    
    # Plot results
    plot_knn_results(results, noise_results)
    
    return results, best_k, noise_results

def plot_knn_results(results, noise_results):
    """Plot KNN performance with and without noise."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy vs k
    k_values = list(results.keys())
    accuracies = [results[k]['accuracy'] for k in k_values]
    
    axes[0].plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Neighbors (k)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('KNN Accuracy vs k (No Noise)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_k]['accuracy']
    axes[0].plot(best_k, best_acc, 'r*', markersize=15, label=f'Best: k={best_k}, acc={best_acc:.3f}')
    axes[0].legend()
    
    # Plot 2: Noise impact
    if noise_results:
        noise_levels = list(noise_results.keys())
        noise_accuracies = [noise_results[nl]['accuracy'] for nl in noise_levels]
        
        axes[1].plot(noise_levels, noise_accuracies, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Noise Level', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('KNN Accuracy vs Noise Level (k=5)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add baseline (no noise)
        baseline_acc = results[5]['accuracy']
        axes[1].axhline(y=baseline_acc, color='blue', linestyle='--', 
                       label=f'Baseline (0 noise): {baseline_acc:.3f}')
        axes[1].legend()
    
    plt.suptitle('KNN Model Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results, best_k, noise_results = train_knn_model_with_noise()
    print("\n✅ KNN model complete with noise experiment!")