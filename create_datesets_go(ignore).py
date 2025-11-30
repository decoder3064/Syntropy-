import pandas as pd
import os

# Original GoEmotions emotion mapping (all 28)
GOEMOTIONS_FULL_MAPPING = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance',
    4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity',
    8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
    12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
    16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
    24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Journal emotions mapped to GoEmotions (overlapping + semantic matches)
JOURNAL_TO_GOEMOTIONS = {
    'happy': 'joy',              # exact semantic match
    'calm': 'neutral',           # semantic match (as you specified)
    'proud': 'pride',            # exact match
    'excited': 'excitement',     # exact match
    'anxious': 'nervousness',    # close semantic match
    'surprised': 'surprise',     # exact match
    'sad': 'sadness',           # exact match
    'angry': 'anger',           # exact match
    'confused': 'confusion',    # exact match
    'disgusted': 'disgust',     # exact match
    'afraid': 'fear',           # exact match
    'awkward': 'embarrassment'  # close semantic match
}

# Get the set of GoEmotions labels we want to keep
OVERLAPPING_EMOTIONS = set(JOURNAL_TO_GOEMOTIONS.values())

print("=" * 70)
print("OVERLAPPING EMOTIONS (Journal ↔ GoEmotions)")
print("=" * 70)
print("\nJournal → GoEmotions Mapping:")
for journal_emo, ge_emo in sorted(JOURNAL_TO_GOEMOTIONS.items()):
    print(f"  {journal_emo:12s} → {ge_emo}")

print(f"\nTotal overlapping emotions: {len(OVERLAPPING_EMOTIONS)}")
print(f"Emotions to keep: {sorted(OVERLAPPING_EMOTIONS)}")

# Create reverse mapping (GoEmotions ID → emotion name)
# but only for emotions we're keeping
FILTERED_ID_TO_LABEL = {}
FILTERED_LABEL_TO_ID = {}
new_id = 0

for old_id, emotion_name in sorted(GOEMOTIONS_FULL_MAPPING.items()):
    if emotion_name in OVERLAPPING_EMOTIONS:
        FILTERED_ID_TO_LABEL[new_id] = emotion_name
        FILTERED_LABEL_TO_ID[emotion_name] = new_id
        new_id += 1

print("\n" + "=" * 70)
print("NEW ID2LABEL MAPPING (for NLP Scholar config)")
print("=" * 70)
print("\nid2label:")
for new_id, emotion in sorted(FILTERED_ID_TO_LABEL.items()):
    print(f"  {new_id}: {emotion}")


def balance_dataset(df, split_name):
    """
    Balance dataset so all emotions have equal representation
    Uses undersampling - samples down to the size of the least frequent emotion
    Preserves column structure (including 'target' for test sets)
    """
    
    print(f"\n{'='*70}")
    print(f"BALANCING {split_name.upper()} DATASET")
    print(f"{'='*70}")
    
    # Determine if this is test set (has 'target' column instead of 'label')
    is_test = 'target' in df.columns
    label_col = 'target' if is_test else 'label'
    
    # Get counts before balancing
    label_counts_before = df[label_col].value_counts()
    print(f"\n Distribution BEFORE balancing:")
    for label, count in label_counts_before.items():
        pct = (count / len(df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    # Find the minimum count
    min_count = label_counts_before.min()
    print(f"\n Target count per emotion: {min_count:,}")
    
    # Sample each emotion to have exactly min_count examples
    balanced_dfs = []
    for emotion in OVERLAPPING_EMOTIONS:
        emotion_df = df[df[label_col] == emotion]
        
        if len(emotion_df) > min_count:
            # Undersample to min_count
            sampled_df = emotion_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled_df)
        else:
            # Keep all examples if already at or below min_count
            balanced_dfs.append(emotion_df)
    
    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset textid to be sequential
    balanced_df['textid'] = range(len(balanced_df))
    
    # Show after balancing
    label_counts_after = balanced_df[label_col].value_counts()
    print(f"\n Distribution AFTER balancing:")
    for label, count in label_counts_after.items():
        pct = (count / len(balanced_df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"\n Reduced from {len(df):,} to {len(balanced_df):,} examples ({len(df) - len(balanced_df):,} removed)")
    
    return balanced_df


def filter_goemotions_by_overlap(input_tsv_path, split_name):
    """
    Filter GoEmotions TSV to only include overlapping emotions
    Returns DataFrame with: text, label (emotion name), source
    For test split: adds text_id column and renames label to target
    """
    
    # Read TSV
    df = pd.read_csv(input_tsv_path, sep='\t', header=None, 
                     names=['text', 'emotion_ids', 'example_id'])
    
    output_rows = []
    skipped_count = 0
    kept_count = 0
    text_id_counter = 0
    
    for idx, row in df.iterrows():
        text = row['text']
        emotion_ids_str = str(row['emotion_ids'])
        
        if emotion_ids_str and emotion_ids_str != 'nan':
            # Parse emotion IDs
            emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
            
            # Filter to only overlapping emotions
            for old_id in emotion_ids:
                emotion_name = GOEMOTIONS_FULL_MAPPING[old_id]
                
                if emotion_name in OVERLAPPING_EMOTIONS:
                    output_rows.append({
                        'textid': text_id_counter,
                        'text': text,
                        'label': emotion_name,
                        'source': 'goemotions'
                    })
                    text_id_counter += 1
                    kept_count += 1
                else:
                    skipped_count += 1
    
    print(f"\n{split_name.upper()} - Kept: {kept_count:,} | Skipped: {skipped_count:,}")
    
    result_df = pd.DataFrame(output_rows)
    
    # For test split, rename 'label' to 'target'
    if split_name.lower() == 'test':
        result_df = result_df.rename(columns={'label': 'target'})
        # Reorder columns: textid, text, target, source
        result_df = result_df[['textid', 'text', 'target', 'source']]
    else:
        # For train/validation: textid, text, label, source
        result_df = result_df[['textid', 'text', 'label', 'source']]
    
    return result_df


def main():
    # ============================================
    # CONFIGURATION - CHANGE THESE PATHS
    # ============================================
    
    # Local GoEmotions input path
    goemotions_input_path = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/'
    
    # Output paths - separate directories for normalized vs unnormalized
    output_unnormalized_path = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_processed_unnormalized/'
    output_normalized_path = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_processed_normalized/'
    
    # REPLACE WITH YOUR TURING USERNAME
    turing_username = 'fdavis'  # ← CHANGE THIS!
    turing_unnormalized_path = f'/home/{turing_username}/data/go_emotions_processed_unnormalized/'
    turing_normalized_path = f'/home/{turing_username}/data/go_emotions_processed_normalized/'
    
    # ============================================
    
    # Create output directories
    os.makedirs(output_unnormalized_path, exist_ok=True)
    os.makedirs(output_normalized_path, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FILTERING GOEMOTIONS DATASETS")
    print("=" * 70)
    
    # Process train, dev, test
    print("\n Loading and filtering train.tsv...")
    train_df = filter_goemotions_by_overlap(f'{goemotions_input_path}train.tsv', 'train')
    
    print("\n Loading and filtering dev.tsv...")
    validate_df = filter_goemotions_by_overlap(f'{goemotions_input_path}dev.tsv', 'validation')
    
    print("\n Loading and filtering test.tsv...")
    test_df = filter_goemotions_by_overlap(f'{goemotions_input_path}test.tsv', 'test')
    
    print("\n" + "=" * 70)
    print("SAVING UNNORMALIZED (FULL) DATASETS")
    print("=" * 70)
    
    # Save UNNORMALIZED (full filtered data)
    train_df.to_csv(f'{output_unnormalized_path}sent_training_data_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_unnormalized_path}sent_training_data_go.tsv")
    
    validate_df.to_csv(f'{output_unnormalized_path}validate_sent_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_unnormalized_path}validate_sent_go.tsv")
    
    test_df.to_csv(f'{output_unnormalized_path}test_sent_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_unnormalized_path}test_sent_go.tsv")
    
    # Balance datasets
    print("\n" + "=" * 70)
    print("BALANCING DATASETS FOR EQUAL EMOTION DISTRIBUTION")
    print("=" * 70)
    
    train_balanced = balance_dataset(train_df, 'train')
    validate_balanced = balance_dataset(validate_df, 'validation')
    test_balanced = balance_dataset(test_df, 'test')
    
    # Save NORMALIZED (balanced)
    print("\n" + "=" * 70)
    print("SAVING NORMALIZED (BALANCED) DATASETS")
    print("=" * 70)
    
    train_balanced.to_csv(f'{output_normalized_path}sent_training_data_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_normalized_path}sent_training_data_go.tsv")
    
    validate_balanced.to_csv(f'{output_normalized_path}validate_sent_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_normalized_path}validate_sent_go.tsv")
    
    test_balanced.to_csv(f'{output_normalized_path}test_sent_go.tsv', sep='\t', index=False)
    print(f" Saved: {output_normalized_path}test_sent_go.tsv")
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS - UNNORMALIZED (FULL)")
    print("=" * 70)
    
    print(f"\n Dataset Sizes (full filtered):")
    print(f"   Training:   {len(train_df):,} examples")
    print(f"   Validation: {len(validate_df):,} examples")
    print(f"   Test:       {len(test_df):,} examples")
    print(f"   Total:      {len(train_df) + len(validate_df) + len(test_df):,} examples")
    
    print("\n Label Distribution in Training Set (unnormalized):")
    label_counts = train_df['label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS - NORMALIZED (BALANCED)")
    print("=" * 70)
    
    print(f"\n Dataset Sizes (balanced):")
    print(f"   Training:   {len(train_balanced):,} examples")
    print(f"   Validation: {len(validate_balanced):,} examples")
    print(f"   Test:       {len(test_balanced):,} examples")
    print(f"   Total:      {len(train_balanced) + len(validate_balanced) + len(test_balanced):,} examples")
    
    print("\n Sample from Training Set (balanced):")
    print(train_balanced.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print(" ALL DATASETS CREATED!")
    print("=" * 70)
    
    print(f"\n UNNORMALIZED output: {output_unnormalized_path}")
    print(f" NORMALIZED output:   {output_normalized_path}")
    
    print("\n Files created:")
    print("\n   UNNORMALIZED (full filtered - larger, imbalanced):")
    print(f"     {output_unnormalized_path}")
    print("       • sent_training_data_go.tsv")
    print("       • validate_sent_go.tsv")
    print("       • test_sent_go.tsv")
    
    print("\n   NORMALIZED (balanced - smaller, equal distribution):")
    print(f"     {output_normalized_path}")
    print("       • sent_training_data_go.tsv")
    print("       • validate_sent_go.tsv")
    print("       • test_sent_go.tsv")
    
    print("\n" + "=" * 70)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n💡 Strategy: Train BOTH models and compare")
    print(f"\n   Model A - UNNORMALIZED ({len(train_df):,} training examples):")
    print("      More data → better representations")
    print("      Real-world emotion distribution")
    print("       Imbalanced → may favor frequent emotions")
    print("      Use class weights or focal loss")
    
    print(f"\n   Model B - NORMALIZED ({len(train_balanced):,} training examples):")
    print("      Equal learning per emotion")
    print("      Fair evaluation across all categories")
    print("       Much less data overall")
    print("      Good baseline for comparison")
    
    print("\n" + "=" * 70)
    print("CONFIG EXAMPLES FOR NLP SCHOLAR")
    print("=" * 70)
    
    print("\n# Config for UNNORMALIZED (full) dataset:")
    print(f"trainfpath: {turing_unnormalized_path}sent_training_data_go.tsv")
    print(f"validfpath: {turing_unnormalized_path}validate_sent_go.tsv")
    print(f"modelfpath: {turing_unnormalized_path}model_goemotions_full")
    
    print("\n# Config for NORMALIZED (balanced) dataset:")
    print(f"trainfpath: {turing_normalized_path}sent_training_data_go.tsv")
    print(f"validfpath: {turing_normalized_path}validate_sent_go.tsv")
    print(f"modelfpath: {turing_normalized_path}model_goemotions_balanced")
    
    print("\n📤 Upload to Turing:")
    print(f"   scp -r {output_unnormalized_path} {turing_username}@turing:/home/{turing_username}/data/")
    print(f"   scp -r {output_normalized_path} {turing_username}@turing:/home/{turing_username}/data/")

if __name__ == "__main__":
    main()
