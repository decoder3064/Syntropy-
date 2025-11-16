import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ============================================
# SPLIT PERCENTAGES - EASY TO MODIFY
# ============================================
# Change these values to control train/validation/test splits
# They should sum to 1.0 (100%)

# ---- UNCOMMENT ONE OF THESE PRESETS OR SET CUSTOM VALUES ----

# ZERO-SHOT: All data goes to test (for evaluation only)
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

# Journal emotions mapped to GoEmotions (overlapping emotions only)
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

# Map journal column names to our mapping
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
    'Answer.f1.disgusted.raw': 'disgusted',  # Fixed: was 'disgust', should be 'disgusted'
    'Answer.f1.afraid.raw': 'afraid',
    'Answer.f1.awkward.raw': 'awkward'
}

def process_journal_dataset(input_csv_path):
    """
    Process journal dataset to extract only overlapping emotions
    Creates one row per emotion (multi-label to multi-row)
    """
    
    print("=" * 70)
    print("PROCESSING JOURNAL DATASET")
    print("=" * 70)
    
    # Read CSV
    df = pd.read_csv(input_csv_path)
    print(f"\n📂 Loaded {len(df)} journal entries")
    
    output_rows = []
    textid_counter = 0
    
    emotion_counts = {emotion: 0 for emotion in JOURNAL_TO_GOEMOTIONS.values()}
    
    for idx, row in df.iterrows():
        text = row['Answer']
        
        # Check each overlapping emotion column
        for col_name, journal_emotion in JOURNAL_COLUMN_MAPPING.items():
            if col_name in df.columns and (row[col_name] == True or row[col_name] == 'TRUE' or row[col_name] == 'true'):
                # Map to GoEmotions emotion name
                goemotions_emotion = JOURNAL_TO_GOEMOTIONS[journal_emotion]
                
                output_rows.append({
                    'textid': textid_counter,
                    'text': text,
                    'label': goemotions_emotion,
                    'source': 'journal'
                })
                textid_counter += 1
                emotion_counts[goemotions_emotion] += 1
    
    result_df = pd.DataFrame(output_rows)
    
    print(f"\n✅ Created {len(result_df)} examples from {len(df)} journal entries")
    print("\n📊 Emotion Distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        pct = (count / len(result_df)) * 100 if len(result_df) > 0 else 0
        print(f"   {emotion:15s}: {count:4,} ({pct:5.2f}%)")
    
    return result_df


def create_splits(df, train_size=None, val_size=None, test_size=None, random_state=42):
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
    
    print("\n" + "=" * 70)
    print("CREATING TRAIN/VALIDATION/TEST SPLITS")
    print("=" * 70)
    print(f"📊 Split configuration: {train_size*100:.0f}% train / {val_size*100:.0f}% val / {test_size*100:.0f}% test")
    
    # Handle zero-shot case (all data goes to test)
    if test_size >= 1.0 or (train_size == 0.0 and val_size == 0.0):
        print("   🎯 Zero-shot mode: All data allocated to test set")
        test_df = df.copy()
        train_df = df.iloc[:0].copy()  # Empty dataframe with same columns
        val_df = df.iloc[:0].copy()     # Empty dataframe with same columns
    else:
        # First split: train vs (val + test)
        # Make sure test_size is valid (< 1.0)
        remaining_size = 1.0 - train_size
        if remaining_size >= 1.0:
            remaining_size = 0.99  # Cap at 0.99 to avoid sklearn error
            
        train_df, temp_df = train_test_split(
            df, 
            test_size=remaining_size,  # Everything except train
            random_state=random_state,
            stratify=df['label']  # Stratify by emotion to maintain distribution
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
            # Of the remaining data, split proportionally between val and test
            relative_test_size = test_size / (val_size + test_size)
            if relative_test_size >= 1.0:
                relative_test_size = 0.99  # Cap at 0.99 to avoid sklearn error
                
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                random_state=random_state,
                stratify=temp_df['label']
            )


def main():
    # ============================================
    # CONFIGURATION - CHANGE THESE PATHS
    # ============================================
    
    # Input journal CSV path
    journal_input_path = '/Users/davidreyes/Documents/Syntropy/data.csv'
    
    # Output path
    output_path = '/Users/davidreyes/Documents/Syntropy/data/journal_processed/'
    
    # REPLACE WITH YOUR TURING USERNAME
    turing_username = 'dreyes'  # ← CHANGE THIS!
    turing_output_path = f'/home/{turing_username}/data/journal_processed/'
    
    # ============================================
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process journal data
    journal_df = process_journal_dataset(journal_input_path)
    
    # Create splits
    train_df, val_df, test_df = create_splits(journal_df)
    
    # Save datasets
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)
    
    train_df.to_csv(f'{output_path}sent_training_data_journal.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_path}sent_training_data_journal.tsv")
    
    val_df.to_csv(f'{output_path}validate_sent_journal.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_path}validate_sent_journal.tsv")
    
    test_df.to_csv(f'{output_path}test_sent_journal.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_path}test_sent_journal.tsv")
    
    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    
    print("\n📝 Sample from Training Set:")
    print(train_df.head(5).to_string(index=False))
    
    print("\n📝 Sample from Test Set (with 'target' column):")
    print(test_df.head(5).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ JOURNAL DATASET PROCESSING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Local output: {output_path}")
    
    print("\n" + "=" * 70)
    print("CONFIG FOR NLP SCHOLAR")
    print("=" * 70)
    
    print("\n# For journal dataset (unnormalized):")
    print(f"trainfpath: {turing_output_path}sent_training_data_journal.tsv")
    print(f"validfpath: {turing_output_path}validate_sent_journal.tsv")
    print(f"modelfpath: {turing_output_path}model_journal")
    
    print("\n📤 Upload to Turing:")
    print(f"   scp -r {output_path} {turing_username}@turing:/home/{turing_username}/data/")
    
    print("\n💡 Dataset Notes:")
    print(f"   • Total entries: {len(journal_df)} (from {len(pd.read_csv(journal_input_path))} journal entries)")
    print("   • This is UNNORMALIZED - emotions have natural distribution")
    print("   • Multi-label entries are expanded to multiple rows")
    print("   • Uses same 12 emotions as filtered GoEmotions")
    print("   • Good for fine-tuning models pretrained on GoEmotions")

if __name__ == "__main__":
    main()