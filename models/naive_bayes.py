

import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
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

def train_naive_bayes_with_noise(noise_levels=[0, 0.05, 0.1, 0.2, 0.3]):

    print("🚀 Training Naive Bayes Model with Noise Experiment")
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
    print(f"Classes: {len(class_names)}")
    
    results = {}
    noise_results = {}
    
    # Train without noise
    print("\n🔧 Training Gaussian Naive Bayes (no noise)...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    y_pred = nb_model.predict(X_test)
    y_pred_proba = nb_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results['no_noise'] = {
        'model': nb_model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'true_labels': y_test,
        'noise_level': 0
    }
    
    print(f"📊 Accuracy (no noise): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Noise experiment
    print(f"\n🔬 Noise Experiment:")
    for noise_level in noise_levels:
        if noise_level == 0:
            continue
            
        print(f"\n📊 Noise Level: {noise_level*100:.0f}%")
        X_train_noisy = add_noise(X_train, noise_level)
        X_test_noisy = add_noise(X_test, noise_level)
        
        nb_noisy = GaussianNB()
        nb_noisy.fit(X_train_noisy, y_train)
        
        y_pred_noisy = nb_noisy.predict(X_test_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        
        noise_results[noise_level] = {
            'model': nb_noisy,
            'accuracy': accuracy_noisy,
            'predictions': y_pred_noisy,
            'true_labels': y_test
        }
        
        accuracy_change = accuracy_noisy - accuracy
        print(f"   Accuracy: {accuracy_noisy:.4f} (Δ = {accuracy_change:+.4f})")
    
    # Plot results
    plot_naive_bayes_results(results['no_noise'], noise_results)
    
    return results, noise_results

def plot_naive_bayes_results(base_result, noise_results):
    """Plot Naive Bayes performance with and without noise."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confusion Matrix (no noise)
    y_true = base_result['true_labels']
    y_pred = base_result['predictions']
    splits = load_splits()
    class_names = splits['class_names']
    
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f'Confusion Matrix (No Noise)\nAccuracy: {base_result["accuracy"]:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    # Plot 2: Noise impact
    if noise_results:
        noise_levels = list(noise_results.keys())
        noise_accuracies = [noise_results[nl]['accuracy'] for nl in noise_levels]
        
        axes[1].plot(noise_levels, noise_accuracies, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Noise Level', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Naive Bayes: Accuracy vs Noise', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add baseline
        baseline_acc = base_result['accuracy']
        axes[1].axhline(y=baseline_acc, color='blue', linestyle='--', 
                       label=f'Baseline (0 noise): {baseline_acc:.3f}')
        axes[1].legend()
        
        # Add percentage labels
        for i, (nl, acc) in enumerate(zip(noise_levels, noise_accuracies)):
            axes[1].annotate(f'{acc:.3f}', (nl, acc), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9)
    
    plt.suptitle('Naive Bayes Model Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results, noise_results = train_naive_bayes_with_noise()
    print("\n✅ Naive Bayes model complete with noise experiment!")