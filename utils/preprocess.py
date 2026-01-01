"""
Preprocessing utilities for color primary prediction project.
Contains functions for labeling colors based on primary composition.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def label_primary_colors(r, g, b, threshold=0.3):
    """
    Label a color based on which primary colors (R, Y, B) are dominant.
    Uses RYB (Red, Yellow, Blue) color model.
    
    Args:
        r, g, b: RGB values (0-255)
        threshold: Minimum normalized value to consider a color present
    
    Returns:
        str: One of 7 primary color combinations
    """
    # Normalize to [0, 1]
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    
    # In RYB model:
    # Yellow = Red + Green (in RGB)
    # We need to determine presence of R, Y, B
    
    # Determine presence of each primary
    has_red = r_norm > threshold and r_norm > b_norm * 0.7
    has_yellow = (r_norm > threshold and g_norm > threshold and b_norm < threshold * 0.5)
    has_blue = b_norm > threshold and b_norm > r_norm * 0.7
    
    # Count how many primaries are present
    primaries = []
    if has_red:
        primaries.append('Red')
    if has_yellow:
        primaries.append('Yellow')
    if has_blue:
        primaries.append('Blue')
    
    # Combine into appropriate class
    if len(primaries) == 0:
        # If no clear primary, it's all three (gray/brown)
        return 'Red+Blue+Yellow'
    elif len(primaries) == 1:
        return primaries[0]
    else:
        # Sort for consistency (alphabetical)
        return '+'.join(sorted(primaries))

def handle_duplicates(df):
    """
    Handle duplicate color names and RGB values.
    Returns cleaned dataframe.
    """
    original_size = len(df)
    
    # Check for duplicate color names
    duplicate_names = df['name'].duplicated().sum()
    if duplicate_names > 0:
        print(f"⚠️  Found {duplicate_names} duplicate color names")
        print("   Adding suffixes to duplicate names...")
        
        # Create a copy of original names
        df['original_name'] = df['name'].copy()
        
        # Add suffixes to duplicates
        def add_suffix(group):
            if len(group) > 1:
                group['name'] = group['original_name'] + '_' + group.groupby('original_name').cumcount().astype(str)
            return group
        
        df = df.groupby('original_name', group_keys=False).apply(add_suffix)
    
    # Check for duplicate RGB combinations
    duplicate_rgb = df.duplicated(subset=['red', 'green', 'blue']).sum()
    if duplicate_rgb > 0:
        print(f"⚠️  Found {duplicate_rgb} duplicate RGB combinations")
        print("   Removing duplicate RGB values (keeping first occurrence)...")
        df = df.drop_duplicates(subset=['red', 'green', 'blue'], keep='first')
    
    removed = original_size - len(df)
    if removed > 0:
        print(f"✅ Removed {removed} duplicates, {len(df)} unique samples remain")
    
    return df

def load_and_label_data(input_path='data/raw/colors.csv', 
                        output_path='data/processed/colors_with_labels.csv'):
    """
    Load raw data, apply labels, and save processed data.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save labeled data
    
    Returns:
        pd.DataFrame: Labeled dataset
    """
    # Check if processed data already exists
    if os.path.exists(output_path):
        print(f"📂 Processed data already exists at {output_path}")
        print("   Loading existing processed data...")
        df = pd.read_csv(output_path)
        print(f"   Loaded {len(df)} labeled samples")
        return df
    
    # Load raw data
    df = pd.read_csv(input_path)
    print(f"📥 Loaded {len(df)} samples from {input_path}")
    
    # Handle duplicates
    df = handle_duplicates(df)
    
    # Apply labels
    print("🏷️  Applying primary color labels...")
    df['primary_label'] = df.apply(
        lambda row: label_primary_colors(row['red'], row['green'], row['blue']), 
        axis=1
    )
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"💾 Saved {len(df)} labeled samples to {output_path}")
    
    # Show class distribution
    print("\n📊 Final Class Distribution:")
    label_counts = df['primary_label'].value_counts()
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        print(f"  {label:20}: {count:5} samples ({percentage:.1f}%)")
    
    # Show imbalance ratio
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n⚖️  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    return df

def prepare_data_for_training(df):
    """
    Prepare features and labels for ML models.
    
    Args:
        df: DataFrame with 'red', 'green', 'blue', 'primary_label' columns
    
    Returns:
        tuple: (X, y, label_encoder)
    """
    # Features: RGB values
    X = df[['red', 'green', 'blue']].values
    
    # Labels: Encode string labels to numbers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['primary_label'])
    
    # Optional: Scale features (0-1 normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoder, scaler

if __name__ == "__main__":
    # Test the preprocessing
    print("🧪 Testing preprocessing functions...")
    print("="*50)
    
    # Test some example colors
    test_colors = [
        (255, 0, 0, 'Red'),
        (0, 0, 255, 'Blue'),
        (255, 255, 0, 'Yellow'),
        (255, 0, 255, 'Red+Blue'),
        (255, 128, 0, 'Red+Yellow'),
        (0, 255, 0, 'Blue+Yellow'),  # Green in RYB
        (128, 128, 128, 'Red+Blue+Yellow')
    ]
    
    print("Testing color labeling logic:")
    for r, g, b, expected in test_colors:
        label = label_primary_colors(r, g, b)
        status = "✅" if label == expected else "❌"
        print(f"{status} RGB({r},{g},{b}) -> {label} (expected: {expected})")
    
    # Load and label the full dataset
    print("\n" + "="*50)
    df = load_and_label_data()
    
    # Show a few samples
    print("\n🔍 Sample of labeled data:")
    print(df[['name', 'red', 'green', 'blue', 'primary_label']].head(10))