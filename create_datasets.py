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
    """
    
    print(f"\n{'='*70}")
    print(f"BALANCING {split_name.upper()} DATASET")
    print(f"{'='*70}")
    
    # Get counts before balancing
    label_counts_before = df['label'].value_counts()
    print(f"\n📊 Distribution BEFORE balancing:")
    for label, count in label_counts_before.items():
        pct = (count / len(df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    # Find the minimum count
    min_count = label_counts_before.min()
    print(f"\n🎯 Target count per emotion: {min_count:,}")
    
    # Sample each emotion to have exactly min_count examples
    balanced_dfs = []
    for emotion in OVERLAPPING_EMOTIONS:
        emotion_df = df[df['label'] == emotion]
        
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
    
    # Show after balancing
    label_counts_after = balanced_df['label'].value_counts()
    print(f"\n📊 Distribution AFTER balancing:")
    for label, count in label_counts_after.items():
        pct = (count / len(balanced_df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"\n✅ Reduced from {len(df):,} to {len(balanced_df):,} examples ({len(df) - len(balanced_df):,} removed)")
    
    return balanced_df


def filter_goemotions_by_overlap(input_tsv_path, split_name):
    """
    Filter GoEmotions TSV to only include overlapping emotions
    Returns DataFrame with: text, label (emotion name), source
    """
    
    # Read TSV
    df = pd.read_csv(input_tsv_path, sep='\t', header=None, 
                     names=['text', 'emotion_ids', 'example_id'])
    
    output_rows = []
    skipped_count = 0
    kept_count = 0
    
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
                        'text': text,
                        'label': emotion_name,
                        'source': 'goemotions'
                    })
                    kept_count += 1
                else:
                    skipped_count += 1
    
    print(f"\n{split_name.upper()} - Kept: {kept_count:,} | Skipped: {skipped_count:,}")
    
    return pd.DataFrame(output_rows)


def main():
    # ============================================
    # CONFIGURATION - CHANGE THESE PATHS
    # ============================================
    
    # Local GoEmotions input path
    goemotions_input_path = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/'
    
    # Output path - saves in data/go_emotions_processed/
    output_base_path = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_processed/'
    
    # REPLACE WITH YOUR TURING USERNAME
    turing_username = 'fdavis'  # ← CHANGE THIS!
    turing_output_path = f'/home/{turing_username}/data/go_emotions_processed/'
    
    # ============================================
    
    # Create output directory
    os.makedirs(output_base_path, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FILTERING GOEMOTIONS DATASETS")
    print("=" * 70)
    
    # Process train, dev, test
    print("\n📂 Loading and filtering train.tsv...")
    train_df = filter_goemotions_by_overlap(f'{goemotions_input_path}train.tsv', 'train')
    
    print("\n📂 Loading and filtering dev.tsv...")
    validate_df = filter_goemotions_by_overlap(f'{goemotions_input_path}dev.tsv', 'validation')
    
    print("\n📂 Loading and filtering test.tsv...")
    test_df = filter_goemotions_by_overlap(f'{goemotions_input_path}test.tsv', 'test')
    
    # Balance datasets
    print("\n" + "=" * 70)
    print("BALANCING DATASETS FOR EQUAL EMOTION DISTRIBUTION")
    print("=" * 70)
    
    train_df = balance_dataset(train_df, 'train')
    validate_df = balance_dataset(validate_df, 'validation')
    test_df = balance_dataset(test_df, 'test')
    
    print("\n" + "=" * 70)
    print("SAVING FILTERED DATASETS")
    print("=" * 70)
    
    # Save with NLP Scholar naming convention
    train_df.to_csv(f'{output_base_path}sent_training_data_go.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_base_path}sent_training_data_go.tsv")
    
    validate_df.to_csv(f'{output_base_path}validate_sent_go.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_base_path}validate_sent_go.tsv")
    
    test_df.to_csv(f'{output_base_path}test_sent_go.tsv', sep='\t', index=False)
    print(f"✅ Saved: {output_base_path}test_sent_go.tsv")
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Filtered Dataset Sizes:")
    print(f"   Training:   {len(train_df):,} examples")
    print(f"   Validation: {len(validate_df):,} examples")
    print(f"   Test:       {len(test_df):,} examples")
    print(f"   Total:      {len(train_df) + len(validate_df) + len(test_df):,} examples")
    
    print("\n📋 Label Distribution in Training Set:")
    label_counts = train_df['label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"   {label:15s}: {count:5,} ({pct:5.2f}%)")
    
    print("\n📝 Sample from Training Set:")
    print(train_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ ALL FILTERED DATASETS CREATED!")
    print("=" * 70)
    print(f"\n📁 Local output: {output_base_path}")
    
    print("\n" + "=" * 70)
    print("CONFIG FOR NLP SCHOLAR (after uploading to Turing)")
    print("=" * 70)
    
    print("\nexp: TextClassification")
    print("\nmode:")
    print("  - train")
    print("\nmodels:")
    print("  hf_text_classification_model:")
    print("    - bert-large-uncased")
    print(f"\ntrainfpath: {turing_output_path}sent_training_data_go.tsv")
    print(f"validfpath: {turing_output_path}validate_sent_go.tsv")
    print(f"modelfpath: {turing_output_path}model_goemotions_filtered")
    print("\ntextLabel: text")
    print("loadPretrained: True")
    print("numLabels:", len(OVERLAPPING_EMOTIONS))
    print("\nid2label:")
    for new_id, emotion in sorted(FILTERED_ID_TO_LABEL.items()):
        print(f"  {new_id}: {emotion}")
    
    print("\n💡 Next Steps:")
    print(f"   1. Upload files to Turing: scp {output_base_path}* {turing_username}@turing:{turing_output_path}")
    print(f"   2. Create directory on Turing: mkdir -p {turing_output_path}")
    print("   3. Update your config file with paths above")
    print("   4. Submit PBS job with GPU queue")

if __name__ == "__main__":
    main()