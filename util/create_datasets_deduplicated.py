import pandas as pd
import os
from sklearn.model_selection import train_test_split

"""
This script creates DEDUPLICATED datasets where each unique text appears only ONCE.
For multi-label examples, it randomly selects ONE emotion label.
This avoids inflating the dataset with duplicate texts.
"""

# ============================================
# CONFIGURATION
# ============================================

# ============================================
# SPLIT PERCENTAGES - EASY TO MODIFY (JOURNAL DATASET ONLY)
# ============================================
# NOTE: These splits only apply to the JOURNAL dataset.
# GoEmotions uses its original pre-split files (train.tsv, dev.tsv, test.tsv)
# 
# Change these values to control train/validation/test splits for Journal
# They should sum to 1.0 (100%)

# ---- UNCOMMENT ONE OF THESE PRESETS OR SET CUSTOM VALUES ----

# ZERO-SHOT: All journal data goes to test (for evaluation only)
TRAIN_SIZE = 0.00
VAL_SIZE = 0.00
TEST_SIZE = 1.00

# FEW-SHOT: Small training set for few-shot learning
# TRAIN_SIZE = 0.05  # 5% for training
# VAL_SIZE = 0.05    # 5% for validation
# TEST_SIZE = 0.90   # 90% for testing

# STANDARD: Typical ML split
# TRAIN_SIZE = 0.70  # 70% for training
# VAL_SIZE = 0.15    # 15% for validation
# TEST_SIZE = 0.15   # 15% for testing

# CUSTOM: Set your own percentages (must sum to 1.0)
# TRAIN_SIZE = 0.10
# VAL_SIZE = 0.10
# TEST_SIZE = 0.80

# ============================================

# GoEmotions emotion mapping
GOEMOTIONS_FULL_MAPPING = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance',
    4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity',
    8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
    12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
    16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
    24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Overlapping emotions between GoEmotions and Journal
OVERLAPPING_EMOTIONS = {'anger', 'confusion', 'disgust', 'embarrassment', 
                        'excitement', 'fear', 'joy', 'nervousness', 
                        'neutral', 'pride', 'sadness', 'surprise'}

# Journal emotion mappings
JOURNAL_TO_GOEMOTIONS = {
    'happy': 'joy',
    'calm': 'neutral',
    'proud': 'pride',
    'excited': 'excitement',
    'anxious': 'nervousness',
    'surprised': 'surprise',
    'sad': 'sadness',
    'angry': 'anger',
    'confused': 'confusion',
    'disgusted': 'disgust',
    'afraid': 'fear',
    'awkward': 'embarrassment'
}

JOURNAL_COLUMN_MAPPING = {
    'Answer.f1.happy.raw': 'happy',
    'Answer.f1.calm.raw': 'calm',
    'Answer.f1.proud.raw': 'proud',
    'Answer.f1.excited.raw': 'excited',
    'Answer.f1.anxious.raw': 'anxious',
    'Answer.f1.surprised.raw': 'surprised',
    'Answer.f1.sad.raw': 'sad',
    'Answer.f1.angry.raw': 'angry',
    'Answer.f1.confused.raw': 'confused',
    'Answer.f1.disgusted.raw': 'disgusted',
    'Answer.f1.afraid.raw': 'afraid',
    'Answer.f1.awkward.raw': 'awkward'
}


def process_goemotions_deduplicated(input_tsv_path, split_name):
    """
    Process GoEmotions TSV: For multi-label examples, randomly pick ONE emotion.
    Each text appears only ONCE in the output.
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING GOEMOTIONS {split_name.upper()} - DEDUPLICATED")
    print(f"{'='*70}")
    
    # Read TSV
    df = pd.read_csv(input_tsv_path, sep='\t', header=None, 
                     names=['text', 'emotion_ids', 'example_id'])
    

    
    output_rows = []
    
    for idx, row in df.iterrows():
        text = row['text']
        emotion_ids_str = str(row['emotion_ids'])
        
        if emotion_ids_str and emotion_ids_str != 'nan':
            # Parse emotion IDs
            emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
            
            # Filter to only overlapping emotions
            overlapping_emotion_names = []
            for eid in emotion_ids:
                emotion_name = GOEMOTIONS_FULL_MAPPING[eid]
                if emotion_name in OVERLAPPING_EMOTIONS:
                    overlapping_emotion_names.append(emotion_name)
            
            # If there are overlapping emotions, randomly pick ONE
            if overlapping_emotion_names:
                # Use index as random seed for reproducibility
                chosen_emotion = pd.Series(overlapping_emotion_names).sample(
                    n=1, random_state=idx
                ).values[0]
                
                output_rows.append({
                    'text': text,
                    'label': chosen_emotion,
                    'source': 'goemotions'
                })
    
    result_df = pd.DataFrame(output_rows)
    
    # Verify no duplicate texts
    duplicates = result_df[result_df.duplicated('text', keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  WARNING: Found {len(duplicates)} duplicate texts!")
        print("First few duplicates:")
        print(duplicates.head())
    else:
        print(f"✅ No duplicate texts found!")
    
    print(f"\n📊 {split_name.upper()} Statistics:")
    print(f"   Total unique texts: {len(result_df):,}")
    
    label_counts = result_df['label'].value_counts()
    print(f"\n📋 Emotion distribution:")
    for emotion, count in label_counts.items():
        pct = (count / len(result_df)) * 100
        print(f"   {emotion:15s}: {count:5,} ({pct:5.2f}%)")
    
    return result_df


def process_journal_deduplicated(input_csv_path):
    """
    Process Journal CSV: For multi-label entries, randomly pick ONE emotion.
    Each text appears only ONCE in the output.
    """
    print(f"\n{'='*70}")
    print("PROCESSING JOURNAL DATASET - DEDUPLICATED")
    print(f"{'='*70}")
    
    # Read CSV
    df = pd.read_csv(input_csv_path)
    print(f"\n📂 Loaded {len(df)} journal entries")
    
    output_rows = []
    
    for idx, row in df.iterrows():
        text = row['Answer']
        
        # Collect all TRUE emotions for this entry
        true_emotions = []
        for col_name, journal_emotion in JOURNAL_COLUMN_MAPPING.items():
            if col_name in df.columns and (row[col_name] == True or row[col_name] == 'TRUE' or row[col_name] == 'true'):
                goemotions_emotion = JOURNAL_TO_GOEMOTIONS[journal_emotion]
                true_emotions.append(goemotions_emotion)
        
        # If there are emotions, randomly pick ONE
        if true_emotions:
            # Use index as random seed for reproducibility
            chosen_emotion = pd.Series(true_emotions).sample(
                n=1, random_state=idx
            ).values[0]
            
            output_rows.append({
                'text': text,
                'label': chosen_emotion,
                'source': 'journal'
            })
    
    result_df = pd.DataFrame(output_rows)
    
    # Verify no duplicate texts
    duplicates = result_df[result_df.duplicated('text', keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  WARNING: Found {len(duplicates)} duplicate texts!")
        print("First few duplicates:")
        print(duplicates.head())
    else:
        print(f"✅ No duplicate texts found!")
    
    print(f"\n📊 Journal Statistics:")
    print(f"   Total unique entries: {len(result_df):,}")
    
    label_counts = result_df['label'].value_counts()
    print(f"\n📋 Emotion distribution:")
    for emotion, count in label_counts.items():
        pct = (count / len(result_df)) * 100
        print(f"   {emotion:15s}: {count:5,} ({pct:5.2f}%)")
    
    return result_df


def create_splits(df, dataset_name, train_size=None, val_size=None, test_size=None, random_state=42):
    """
    Create train/validation/test splits
    Uses global configuration values if not specified
    """
    # Use global config if not specified
    if train_size is None:
        train_size = TRAIN_SIZE
    if val_size is None:
        val_size = VAL_SIZE
    if test_size is None:
        test_size = TEST_SIZE
    
    print(f"\n{'='*70}")
    print(f"CREATING SPLITS FOR {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"📊 Split configuration: {train_size*100:.0f}% train / {val_size*100:.0f}% val / {test_size*100:.0f}% test")
    
    # Handle zero-shot case (all data goes to test)
    if test_size >= 1.0 or (train_size == 0.0 and val_size == 0.0):
        print("   🎯 Zero-shot mode: All data allocated to test set")
        test_df = df.copy()
        train_df = df.iloc[:0].copy()  # Empty dataframe with same columns
        val_df = df.iloc[:0].copy()     # Empty dataframe with same columns
    else:
        # First split: train vs (val + test)
        remaining_size = 1.0 - train_size
        if remaining_size >= 1.0:
            remaining_size = 0.99  # Cap to avoid sklearn error
            
        train_df, temp_df = train_test_split(
            df, 
            test_size=remaining_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        # Second split: validation vs test
        if val_size == 0.0:
            # No validation set, all temp goes to test
            val_df = temp_df.iloc[:0].copy()
            test_df = temp_df.copy()
        elif test_size == 0.0:
            # No test set, all temp goes to validation
            test_df = temp_df.iloc[:0].copy()
            val_df = temp_df.copy()
        else:
            relative_test_size = test_size / (val_size + test_size)
            if relative_test_size >= 1.0:
                relative_test_size = 0.99  # Cap to avoid sklearn error
                
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                random_state=random_state,
                stratify=temp_df['label']
            )
    
    # Add textid AFTER splitting
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['textid'] = range(len(train_df))
    val_df['textid'] = range(len(val_df))
    test_df['textid'] = range(len(test_df))
    
    # Reorder columns: textid, text, label, source
    train_df = train_df[['textid', 'text', 'label', 'source']]
    val_df = val_df[['textid', 'text', 'label', 'source']]
    test_df = test_df[['textid', 'text', 'label', 'source']]
    
    # Rename 'label' to 'target' for test set
    test_df = test_df.rename(columns={'label': 'target'})
    
    print(f"\n📊 Split sizes:")
    print(f"   Training:   {len(train_df):4,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df):4,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:       {len(test_df):4,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Total:      {len(df):4,}")
    
    return train_df, val_df, test_df


def main():
    # ============================================
    # PATHS CONFIGURATION
    # ============================================
    
    # GoEmotions input
    goemotions_input_path = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/'
    
    # Journal input
    journal_input_path = '/Users/davidreyes/Documents/Syntropy/data.csv'
    
    # Output paths
    go_output_path = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_deduplicated/'
    journal_output_path = '/Users/davidreyes/Documents/Syntropy/data/journal_deduplicated/'
    
    # Create output directories
    os.makedirs(go_output_path, exist_ok=True)
    os.makedirs(journal_output_path, exist_ok=True)
    
    print("="*70)
    print("DEDUPLICATED DATASET CREATION")
    print("="*70)
    print("\n🎯 Goal: One text = One label (no inflation)")
    print("   Multi-label texts: Randomly pick ONE emotion")
    print("   Result: Each unique text appears exactly ONCE")
    
    # ============================================
    # PROCESS GOEMOTIONS
    # ============================================
    
    print("\n" + "="*70)
    print("PROCESSING GOEMOTIONS DATASETS")
    print("="*70)
    print("ℹ️  NOTE: GoEmotions uses its original pre-split files")
    print("   (train.tsv, dev.tsv, test.tsv) - not affected by config above")
    
    train_go = process_goemotions_deduplicated(f'{goemotions_input_path}train.tsv', 'train')
    val_go = process_goemotions_deduplicated(f'{goemotions_input_path}dev.tsv', 'validation')
    test_go = process_goemotions_deduplicated(f'{goemotions_input_path}test.tsv', 'test')
    
    # Add textid and save
    train_go['textid'] = range(len(train_go))
    val_go['textid'] = range(len(val_go))
    test_go['textid'] = range(len(test_go))
    
    # Reorder columns
    train_go = train_go[['textid', 'text', 'label', 'source']]
    val_go = val_go[['textid', 'text', 'label', 'source']]
    test_go = test_go[['textid', 'text', 'label', 'source']]
    
    # Rename label to target for test
    test_go = test_go.rename(columns={'label': 'target'})
    
    # Save GoEmotions
    train_go.to_csv(f'{go_output_path}sent_training_data_go.tsv', sep='\t', index=False)
    val_go.to_csv(f'{go_output_path}validate_sent_go.tsv', sep='\t', index=False)
    test_go.to_csv(f'{go_output_path}test_sent_go.tsv', sep='\t', index=False)
    
    print(f"\n✅ Saved GoEmotions to: {go_output_path}")
    
    # ============================================
    # PROCESS JOURNAL
    # ============================================
    
    print("\n" + "="*70)
    print("PROCESSING JOURNAL DATASET")
    print("="*70)
    print(f"ℹ️  Using configured split: {TRAIN_SIZE*100:.0f}% train / {VAL_SIZE*100:.0f}% val / {TEST_SIZE*100:.0f}% test")
    
    journal_df = process_journal_deduplicated(journal_input_path)
    
    # Create splits (uses TRAIN_SIZE, VAL_SIZE, TEST_SIZE from config)
    train_j, val_j, test_j = create_splits(journal_df, 'journal')
    
    # Save Journal
    train_j.to_csv(f'{journal_output_path}sent_training_data_journal.tsv', sep='\t', index=False)
    val_j.to_csv(f'{journal_output_path}validate_sent_journal.tsv', sep='\t', index=False)
    test_j.to_csv(f'{journal_output_path}test_sent_journal.tsv', sep='\t', index=False)
    
    print(f"\n✅ Saved Journal to: {journal_output_path}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY - DEDUPLICATED DATASETS")
    print("="*70)
    
    print("\n GoEmotions (deduplicated):")
    print(f"   Training:   {len(train_go):5,} unique texts")
    print(f"   Validation: {len(val_go):5,} unique texts")
    print(f"   Test:       {len(test_go):5,} unique texts")
    print(f"   Total:      {len(train_go) + len(val_go) + len(test_go):5,} unique texts")
    
    print("\n Journal (deduplicated):")
    print(f"   Training:   {len(train_j):5,} unique texts")
    print(f"   Validation: {len(val_j):5,} unique texts")
    print(f"   Test:       {len(test_j):5,} unique texts")
    print(f"   Total:      {len(train_j) + len(val_j) + len(test_j):5,} unique texts")
    
    print("\n" + "="*70)
    print("COMPARISON: Original vs Deduplicated")
    print("="*70)
    
    print("\n📈 GoEmotions Training Set:")
    print("   Original (inflated):      23,812 rows")
    print(f"   Deduplicated (this run):  {len(train_go):5,} rows")
    print(f"   Reduction:                {23812 - len(train_go):5,} rows removed")
    
    print("\n📈 Journal Total:")
    print("   Original (inflated):      2,029 rows")
    print(f"   Deduplicated (this run):  {len(journal_df):5,} rows")
    print(f"   Reduction:                {2029 - len(journal_df):5,} rows removed")
    
    print("\n" + "="*70)
    print("✅ DEDUPLICATED DATASETS CREATED SUCCESSFULLY!")
    print("="*70)
    
    print("\n💡 Key points:")
    print("   ✓ Each text appears exactly ONCE")
    print("   ✓ Multi-label texts: randomly picked ONE emotion")
    print("   ✓ No duplicate texts with different labels")
    print("   ✓ Stratified splits maintain emotion distribution")
    
    print(f"\n📁 Output locations:")
    print(f"   GoEmotions: {go_output_path}")
    print(f"   Journal:    {journal_output_path}")


if __name__ == "__main__":
    main()
