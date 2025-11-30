import pandas as pd
import os
from sklearn.model_selection import train_test_split

"""
This script creates train/validation/test splits for the balanced journal dataset.
Supports zero-shot, few-shot, and standard training scenarios.
"""

# ============================================
# SPLIT CONFIGURATIONS
# ============================================
# This script will create 3 different splits automatically:
# 1. Zero-shot: 0% train, 100% test
# 2. 10% Few-shot: 10% train, 90% test
# 3. 35% Few-shot: 35% train, 65% test

SPLIT_CONFIGS = [
    {
        'name': 'zeroshot',
        'train': 0.00,
        'test': 1.00
    },
    {
        'name': 'fewshot_10pct',
        'train': 0.10,
        'test': 0.90
    },
    {
        'name': 'fewshot_35pct',
        'train': 0.35,
        'test': 0.65
    }
]

# ============================================


def create_splits(df, train_size=None, test_size=None, random_state=42):
    """
    Create train/test splits using sklearn train_test_split.
    
    Args:
        df: DataFrame with columns [textid, text, label, source]
        train_size: Fraction for training
        test_size: Fraction for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df
    """
    print(f"\n Split configuration: {train_size*100:.0f}% train / {test_size*100:.0f}% test")
    
    # Initialize empty dataframes
    train_df = df.iloc[:0].copy()
    test_df = df.iloc[:0].copy()
    
    # Handle edge cases
    if train_size == 0.0:
        # Zero-shot: all to test
        print(" Zero-shot mode: All data allocated to test set")
        test_df = df.copy()
    elif test_size == 0.0:
        # All to training
        train_df = df.copy()
    else:
        # Standard train/test split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label']
        )
    
    # Reset textid for each split
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['textid'] = range(len(train_df))
    test_df['textid'] = range(len(test_df))
    
    # Rename 'label' to 'target' for test set
    if len(test_df) > 0:
        test_df = test_df.rename(columns={'label': 'target'})
    
    print(f"\n Split sizes:")
    print(f"   Training: {len(train_df):4,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Test:     {len(test_df):4,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Total:    {len(df):4,}")
    
    # Show emotion distribution in each split
    if len(train_df) > 0:
        print("\n Emotion distribution in TRAIN:")
        train_dist = train_df['label'].value_counts()
        for emotion, count in train_dist.items():
            pct = (count / len(train_df)) * 100
            print(f"   {emotion:15s}: {count:3,} ({pct:5.2f}%)")
    
    label_col = 'target' if 'target' in test_df.columns else 'label'
    if len(test_df) > 0:
        print(f"\n Emotion distribution in TEST:")
        test_dist = test_df[label_col].value_counts()
        for emotion, count in test_dist.items():
            pct = (count / len(test_df)) * 100
            print(f"   {emotion:15s}: {count:3,} ({pct:5.2f}%)")
    
    return train_df, test_df


def main():
    """
    Main function to create multiple train/test splits for journal dataset.
    Creates 3 different configurations: zero-shot, 10% few-shot, and 35% few-shot.
    """
    
    print("="*70)
    print("JOURNAL DATASET TRAIN/TEST SPLIT CREATION")
    print("="*70)
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Input: Use data.csv to create fresh journal dataset
    journal_csv = '/Users/davidreyes/Documents/Syntropy/data.csv'
    
    # Output directory - all splits saved here
    output_dir = '/Users/davidreyes/Documents/Syntropy/data/zero_and_fewshot/'
    
    # ============================================
    
    print(f"\n📁 Input file: {journal_csv}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n🎯 Creating {len(SPLIT_CONFIGS)} different split configurations")
    
    # Check if input exists
    if not os.path.exists(journal_csv):
        print(f"\n⚠️  ERROR: Input file not found: {journal_csv}")
        print("\n💡 Make sure data.csv exists")
        return
    
    # Import the processing function
    import sys
    sys.path.append('/Users/davidreyes/Documents/Syntropy')
    from util.create_datasets_deduplicated import process_journal_deduplicated
    
    # Process journal data
    print(f"\n📂 Processing journal data...")
    df = process_journal_deduplicated(journal_csv)
    
    if len(df) == 0:
        print("\n⚠️  ERROR: No journal data was processed")
        return
    
    print(f"✅ Processed {len(df)} journal entries")
    
    # Verify columns
    required_cols = ['text', 'label', 'source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n  ERROR: Missing required columns: {missing_cols}")
        print(f"   Found columns: {list(df.columns)}")
        return
    
    # Show original distribution
    print("\n Original emotion distribution:")
    label_counts = df['label'].value_counts()
    for emotion, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {emotion:15s}: {count:4,} ({pct:5.2f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split configuration
    for config in SPLIT_CONFIGS:
        print("\n" + "=" * 70)
        print(f"CREATING {config['name'].upper()} SPLIT")
        print("=" * 70)
        
        # Create splits
        train_df, test_df = create_splits(
            df, 
            train_size=config['train'],
            test_size=config['test']
        )
        
        # Create subfolder for this configuration
        config_dir = os.path.join(output_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        # Save datasets in subfolder with config name in filename
        if len(train_df) > 0:
            train_file = f'{config_dir}/train_{config["name"]}_journal.tsv'
            train_df.to_csv(train_file, sep='\t', index=False)
            print(f" Saved: {train_file}")
        
        test_file = f'{config_dir}/test_{config["name"]}_journal.tsv'
        test_df.to_csv(test_file, sep='\t', index=False)
        print(f" Saved: {test_file}")
        
        # Show sample from this split
        if len(train_df) > 0:
            print(f"\n Sample from {config['name']} training set:")
            print(train_df.head(2).to_string(index=False))
    
    print("\n" + "=" * 70)
    print(" ALL JOURNAL DATASET SPLITS COMPLETE!")
    print("=" * 70)
    
    print("\n Created files:")
    for config in SPLIT_CONFIGS:
        print(f"\n   {config['name'].upper()} ({config['train']*100:.0f}% train / {config['test']*100:.0f}% test):")
        print(f"      {output_dir}{config['name']}/")
        if config['train'] > 0:
            print(f"         • train_{config['name']}_journal.tsv")
        print(f"         • test_{config['name']}_journal.tsv")
    
    print("\n All files saved to:", output_dir)


if __name__ == "__main__":
    main()
