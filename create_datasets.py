import pandas as pd
import os

# Emotion ID to name mapping
EMOTION_MAPPING = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance',
    4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity',
    8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
    12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
    16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
    24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

def convert_goemotions_to_format(input_tsv_path):
    """
    Convert GoEmotions TSV to desired format
    Returns a DataFrame with columns: text, label, source
    
    For multi-label examples, creates multiple rows (one per emotion)
    """
    
    # Read TSV with correct column order
    df = pd.read_csv(input_tsv_path, sep='\t', header=None, 
                     names=['text', 'emotion_ids', 'example_id'])
    
    output_rows = []
    
    for idx, row in df.iterrows():
        text = row['text']
        emotion_ids_str = str(row['emotion_ids'])
        
        # Parse comma-separated emotion IDs
        if emotion_ids_str and emotion_ids_str != 'nan':
            # Handle both single IDs and comma-separated IDs
            emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
            
            # Create one row per emotion label
            for eid in emotion_ids:
                emotion_name = EMOTION_MAPPING[eid]
                output_rows.append({
                    'text': text,
                    'label': emotion_name,
                    'source': 'goemotions'
                })
    
    return pd.DataFrame(output_rows)

def main():
    # Paths
    goemotions_path = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/'
    output_base_path = '/Users/davidreyes/Documents/Syntropy/data/'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    print("=" * 70)
    print("Converting GoEmotions datasets")
    print("=" * 70)
    
    # Load train, dev, test
    print("\n📂 Loading train.tsv...")
    train_df = convert_goemotions_to_format(f'{goemotions_path}train.tsv')
    print(f"   ✅ Loaded {len(train_df)} training examples")
    
    print("\n📂 Loading dev.tsv...")
    validate_df = convert_goemotions_to_format(f'{goemotions_path}dev.tsv')
    print(f"   ✅ Loaded {len(validate_df)} validation examples")
    
    print("\n📂 Loading test.tsv...")
    test_df = convert_goemotions_to_format(f'{goemotions_path}test.tsv')
    print(f"   ✅ Loaded {len(test_df)} test examples")
    
    print("\n" + "=" * 70)
    print("Saving datasets")
    print("=" * 70)
    
    # Save as TSV (tab-separated)
    print("\n💾 Saving as TSV format...")
    train_df.to_csv(f'{output_base_path}train.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}train.tsv")
    
    validate_df.to_csv(f'{output_base_path}validate.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}validate.tsv")
    
    test_df.to_csv(f'{output_base_path}test.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}test.tsv")
    
    print("\n" + "=" * 70)
    print("Summary & Sample Data")
    print("=" * 70)
    
    print("\n📊 Dataset Statistics:")
    print(f"   Training:   {len(train_df):,} examples")
    print(f"   Validation: {len(validate_df):,} examples")
    print(f"   Test:       {len(test_df):,} examples")
    print(f"   Total:      {len(train_df) + len(validate_df) + len(test_df):,} examples")
    
    print("\n📋 Label Distribution in Training Set:")
    label_counts = train_df['label'].value_counts()
    for label, count in label_counts.head(10).items():
        print(f"   {label:15s}: {count:,}")
    
    print("\n📝 Sample from Training Set:")
    print(train_df.head(5).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ ALL DATASETS CREATED AND SAVED!")
    print("=" * 70)
    print(f"\n📁 Output location: {output_base_path}")
    print("\nFiles created:")
    print("   • train.tsv      - Training set (original GoEmotions train)")
    print("   • validate.tsv   - Validation set (original GoEmotions dev)")
    print("   • test.tsv       - Test set (original GoEmotions test)")

if __name__ == "__main__":
    main()
