"""
Create consistent train/test splits for all models.
Ensures fair comparison between different algorithms.
Now includes balance checking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def check_class_balance(y_train, y_test, label_encoder):
    """Check and report class distribution balance."""
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    print("\n📊 Class Balance Analysis:")
    print("="*40)
    
    for i, class_name in enumerate(label_encoder.classes_):
        train_percent = train_counts[i] / len(y_train) * 100
        test_percent = test_counts[i] / len(y_test) * 100
        print(f"{class_name:20}: Train={train_counts[i]:4} ({train_percent:5.1f}%) | "
              f"Test={test_counts[i]:4} ({test_percent:5.1f}%)")
    
    # Calculate imbalance ratio
    train_imbalance = max(train_counts) / min(train_counts)
    test_imbalance = max(test_counts) / min(test_counts)
    
    print(f"\n⚖️  Imbalance Ratio:")
    print(f"   Training set: {train_imbalance:.1f}:1")
    print(f"   Test set: {test_imbalance:.1f}:1")
    
    if train_imbalance > 10 or test_imbalance > 10:
        print("⚠️  Warning: Severe class imbalance detected!")
        print("   Consider using class_weight='balanced' in models")
    else:
        print("✅ Class distribution is acceptable")

def create_train_test_splits():
    """Create and save consistent train/test splits."""
    
    # Load labeled data
    data_path = 'data/processed/colors_with_labels.csv'
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} labeled samples")
    
    # Prepare features and labels
    X = df[['red', 'green', 'blue']].values
    y = df['primary_label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data (80% train, 20% test) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Create scaled versions
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create splits directory if it doesn't exist
    splits_dir = 'data/processed/splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    # Save all splits
    splits = {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'class_names': label_encoder.classes_,
        'class_weights': dict(zip(
            label_encoder.classes_,
            len(y_train) / (len(label_encoder.classes_) * np.bincount(y_train))
        ))
    }
    
    with open(f'{splits_dir}/splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    # Also save as CSV for manual inspection
    train_df = pd.DataFrame(X_train, columns=['red', 'green', 'blue'])
    train_df['label'] = label_encoder.inverse_transform(y_train)
    train_df['split'] = 'train'
    
    test_df = pd.DataFrame(X_test, columns=['red', 'green', 'blue'])
    test_df['label'] = label_encoder.inverse_transform(y_test)
    test_df['split'] = 'test'
    
    combined_df = pd.concat([train_df, test_df])
    combined_df.to_csv(f'{splits_dir}/splits.csv', index=False)
    
    print("\n✅ Train/test splits created and saved!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Saved to: {splits_dir}/")
    print(f"   Classes: {', '.join(label_encoder.classes_)}")
    
    # Show class distribution
    check_class_balance(y_train, y_test, label_encoder)
    
    # Calculate and show class weights for balanced training
    print("\n📈 Suggested class weights for balanced training:")
    class_weights = splits['class_weights']
    for class_name, weight in class_weights.items():
        print(f"   {class_name:20}: {weight:.3f}")

if __name__ == "__main__":
    create_train_test_splits()
    