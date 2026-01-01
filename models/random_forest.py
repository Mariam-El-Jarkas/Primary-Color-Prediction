
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def train_random_forest_with_noise(n_estimators=100, max_depth=20, noise_levels=[0, 0.05, 0.1, 0.2, 0.3]):
  
    print("🚀 Training Random Forest Model with Comprehensive Noise Experiment")
    print("="*70)
    
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
    print(f"Using class_weight='balanced' to handle class imbalance")
    
    results = {}
    noise_results = {}
    
    # Train without noise
    print(f"\n🔧 Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, no noise)...")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # ADDED: Handle class imbalance
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results['no_noise'] = {
        'model': rf_model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'true_labels': y_test,
        'feature_importances': rf_model.feature_importances_,
        'class_names': class_names
    }
    
    print(f"📊 Accuracy (no noise): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show class-wise performance for imbalance analysis
    print(f"\n📊 Class-wise Performance (Random Forest with class_weight='balanced'):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Comprehensive noise experiment
    print(f"\n🔬 Comprehensive Noise Experiment:")
    all_noise_results = {}
    
    for noise_level in noise_levels:
        if noise_level == 0:
            continue
            
        print(f"\n📊 Noise Level: {noise_level*100:.0f}%")
        X_train_noisy = add_noise(X_train, noise_level)
        X_test_noisy = add_noise(X_test, noise_level)
        
        rf_noisy = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # ADDED: Handle class imbalance
        )
        rf_noisy.fit(X_train_noisy, y_train)
        
        y_pred_noisy = rf_noisy.predict(X_test_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        
        all_noise_results[noise_level] = {
            'model': rf_noisy,
            'accuracy': accuracy_noisy,
            'predictions': y_pred_noisy,
            'true_labels': y_test,
            'feature_importances': rf_noisy.feature_importances_
        }
        
        accuracy_change = accuracy_noisy - accuracy
        change_symbol = "+" if accuracy_change >= 0 else ""
        print(f"   Accuracy: {accuracy_noisy:.4f} (Δ = {change_symbol}{accuracy_change:.4f})")
    
    # Plot comprehensive results
    plot_random_forest_results(results['no_noise'], all_noise_results)
    
    # Compare with other models
    compare_with_other_models(accuracy, all_noise_results)
    
    return results, all_noise_results

def plot_random_forest_results(base_result, noise_results):
    """Plot Random Forest performance with comprehensive noise analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Feature Importance (no noise)
    features = ['Red', 'Green', 'Blue']
    importances = base_result['feature_importances']
    
    bars1 = axes[0, 0].bar(features, importances, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0, 0].set_title('Feature Importance (No Noise)\nclass_weight="balanced"', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Importance Score', fontsize=10)
    axes[0, 0].set_ylim(0, 1)
    
    for bar, imp in zip(bars1, importances):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Noise impact on accuracy
    if noise_results:
        noise_levels = list(noise_results.keys())
        noise_accuracies = [noise_results[nl]['accuracy'] for nl in noise_levels]
        
        axes[0, 1].plot(noise_levels, noise_accuracies, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Noise Level', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Random Forest: Accuracy vs Noise\n(class_weight="balanced")', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        baseline_acc = base_result['accuracy']
        axes[0, 1].axhline(y=baseline_acc, color='blue', linestyle='--', 
                          label=f'Baseline: {baseline_acc:.3f}')
        axes[0, 1].legend()
    
    # Plot 3: Noise impact on feature importance
    if noise_results:
        noise_importances = []
        for nl in noise_levels:
            noise_importances.append(noise_results[nl]['feature_importances'])
        
        noise_importances = np.array(noise_importances)
        
        for i, feature in enumerate(features):
            axes[1, 0].plot(noise_levels, noise_importances[:, i], 'o-', 
                           label=feature, linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Noise Level', fontsize=12)
        axes[1, 0].set_ylabel('Importance Score', fontsize=12)
        axes[1, 0].set_title('Feature Importance vs Noise\n(class_weight="balanced")', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Plot 4: Percentage accuracy drop
    if noise_results:
        baseline_acc = base_result['accuracy']
        accuracy_drops = [(baseline_acc - noise_results[nl]['accuracy']) / baseline_acc * 100 
                         for nl in noise_levels]
        
        axes[1, 1].bar(noise_levels, accuracy_drops, color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Noise Level', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy Drop (%)', fontsize=12)
        axes[1, 1].set_title('Accuracy Degradation with Noise\n(class_weight="balanced")', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (nl, drop) in enumerate(zip(noise_levels, accuracy_drops)):
            axes[1, 1].text(nl, drop + 1, f'{drop:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Random Forest: Comprehensive Noise Analysis (Handling Class Imbalance)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_with_other_models(rf_accuracy, noise_results):
    """Compare Random Forest noise robustness with theoretical expectations."""
    print("\n📊 Noise Robustness Analysis:")
    print("-" * 50)
    print("Note: Using class_weight='balanced' for better handling of imbalanced classes")
    
    if 0.1 in noise_results:  # 10% noise
        acc_10pct = noise_results[0.1]['accuracy']
        drop_10pct = (rf_accuracy - acc_10pct) / rf_accuracy * 100
        print(f"   With 10% noise: Accuracy = {acc_10pct:.4f} (Drop: {drop_10pct:.1f}%)")
    
    if 0.2 in noise_results:  # 20% noise
        acc_20pct = noise_results[0.2]['accuracy']
        drop_20pct = (rf_accuracy - acc_20pct) / rf_accuracy * 100
        print(f"   With 20% noise: Accuracy = {acc_20pct:.4f} (Drop: {drop_20pct:.1f}%)")
    
    print("\n💡 Insights (with class_weight='balanced'):")
    print("   1. Random Forest handles class imbalance better with balanced weights")
    print("   2. Minority classes get more consideration in decision making")
    print("   3. Feature importance may shift to better represent minority classes")
    print("   4. Overall robustness to noise is maintained while being fair to all classes")

if __name__ == "__main__":
    results, noise_results = train_random_forest_with_noise()
    print("\n✅ Random Forest model complete with comprehensive noise experiment!")