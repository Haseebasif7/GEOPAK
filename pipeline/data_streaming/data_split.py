"""
Dataset splitting utilities for train/test split
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test_split(csv_path, test_size=0.1, random_state=42, output_dir=None):
    """
    Create train/test split (90% train, 10% test)
    
    Args:
        csv_path: Path to full dataset CSV
        test_size: Fraction for test set (default 0.1 = 10%)
        random_state: Random seed for reproducibility
        output_dir: Directory to save split CSVs (if None, uses same dir as csv_path)
    
    Returns:
        train_df, test_df: DataFrames for train and test sets
    """
    print("=" * 70)
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # Load full dataset
    df = pd.read_csv(csv_path)
    print(f"\n📊 Full dataset: {len(df):,} samples")
    
    # Show province distribution
    if 'province' in df.columns:
        print(f"\n📍 Province distribution (full dataset):")
        province_counts = df['province'].value_counts().sort_index()
        for province, count in province_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {province:<25} {count:>6,} ({pct:>5.2f}%)")
    
    # Stratified split by province to maintain distribution
    if 'province' in df.columns:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['province']  # Maintain province distribution
        )
    else:
        # Simple random split if no province column
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Show province distribution in splits
    if 'province' in df.columns:
        print(f"\n📍 Train set province distribution:")
        train_counts = train_df['province'].value_counts().sort_index()
        for province, count in train_counts.items():
            pct = (count / len(train_df)) * 100
            print(f"   {province:<25} {count:>6,} ({pct:>5.2f}%)")
        
        print(f"\n📍 Test set province distribution:")
        test_counts = test_df['province'].value_counts().sort_index()
        for province, count in test_counts.items():
            pct = (count / len(test_df)) * 100
            print(f"   {province:<25} {count:>6,} ({pct:>5.2f}%)")
    
    # Save split CSVs
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n💾 Saved splits:")
    print(f"   Train: {train_csv}")
    print(f"   Test:  {test_csv}")
    
    return train_df, test_df


if __name__ == '__main__':
    # Example usage
    csv_path = 'final.csv'
    train_df, test_df = create_train_test_split(csv_path, test_size=0.1, random_state=42)
