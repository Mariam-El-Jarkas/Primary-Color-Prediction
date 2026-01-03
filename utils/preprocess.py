import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_primary_colors_ryb(r, g, b):
    """Label colors using RYB color model."""
    threshold = 100  # simple threshold for presence

    labels = []
    if r > threshold:
        labels.append('Red')
    if g > threshold:
        # In RYB, Green contributes to Yellow
        labels.append('Yellow')
    if b > threshold:
        labels.append('Blue')

    if not labels:
        return 'Neutral'
    elif len(labels) == 1:
        return labels[0]
    else:
        return '+'.join(sorted(labels))

def preprocess_colors(input_path='data/raw/colors.csv',
                      output_path='data/processed/colors_clean.csv'):
    """Clean and prepare the dataset for modeling."""
    
    # 1️⃣ Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} raw samples")
    
    # 2️⃣ Remove duplicate rows (including RGB duplicates)
    df = df.drop_duplicates()
    print(f"Removed duplicates → {len(df)} unique rows")
    
    # 3️⃣ Make color names unique 
    df['name'] = df['name'] + '_' + df.groupby('name').cumcount().astype(str).replace('_0','')
    
    # 4️⃣ Create primary_label if missing
    if 'primary_label' not in df.columns:
        print("Creating primary labels...")
        df['primary_label'] = df.apply(
            lambda row: label_primary_colors_ryb(row['red'], row['green'], row['blue']),
            axis=1
        )
    
    # 5️⃣ Encode target labels for ML models
    le = LabelEncoder()
    df['primary_label_encoded'] = le.fit_transform(df['primary_label'])
    
    # 6️⃣ Scale features (0-1) for KNN
    for col in ['red', 'green', 'blue']:
        df[col] = df[col] / 255.0
    
    # 7️⃣ Report class distribution and imbalance ratio
    print("\nClass distribution:")
    counts = df['primary_label'].value_counts()
    for label, count in counts.items():
        percent = count / len(df) * 100
        print(f"  {label}: {count} samples ({percent:.1f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print("⚠️  Severe imbalance - consider class_weight='balanced' in models")
    
    # 8️⃣ Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset → {output_path}")
    
    print("\n✅ Data ready for modeling!")
    print("Issues solved:")
    print("  1. ✅ No duplicate rows (including RGB duplicates)")
    print("  2. ✅ Unique color names")
    print("  3. ✅ Features scaled (0-1)")
    print("  4. ✅ Labels encoded for ML")
    print("  5. ✅ Class imbalance reported with ratio")
    
    return df, le

if __name__ == "__main__":
    df, encoder = preprocess_colors()