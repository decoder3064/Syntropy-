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
    # ============================================
    # CONFIGURATION - CHANGE THESE PATHS
    # ============================================
    
    # Local GoEmotions input path
    goemotions_input_path = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/'
    
    # Output path - saves in data/go_emotions_processed/
    output_base_path = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_processed/'
    
    # REPLACE 'username' WITH YOUR ACTUAL TURING USERNAME
    turing_username = 'username'  # ← CHANGE THIS!
    turing_output_path = f'/home/{turing_username}/data/go_emotions_processed/'
    
    # ============================================
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    print("=" * 70)
    print("Converting GoEmotions datasets")
    print("=" * 70)
    
    # Load train, dev, test
    print("\n📂 Loading train.tsv...")
    train_df = convert_goemotions_to_format(f'{goemotions_input_path}train.tsv')
    print(f"   ✅ Loaded {len(train_df)} training examples")
    
    print("\n📂 Loading dev.tsv...")
    validate_df = convert_goemotions_to_format(f'{goemotions_input_path}dev.tsv')
    print(f"   ✅ Loaded {len(validate_df)} validation examples")
    
    print("\n📂 Loading test.tsv...")
    test_df = convert_goemotions_to_format(f'{goemotions_input_path}test.tsv')
    print(f"   ✅ Loaded {len(test_df)} test examples")
    
    print("\n" + "=" * 70)
    print("Saving datasets with NLP Scholar naming convention")
    print("=" * 70)
    
    # Save with NLP Scholar naming convention
    print("\n💾 Saving as TSV format...")
    
    train_df.to_csv(f'{output_base_path}sent_training_data_go.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}sent_training_data_go.tsv")
    
    validate_df.to_csv(f'{output_base_path}validate_sent_go.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}validate_sent_go.tsv")
    
    test_df.to_csv(f'{output_base_path}test_sent_go.tsv', sep='\t', index=False)
    print(f"   ✅ Saved: {output_base_path}test_sent_go.tsv")
    
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
    print(f"\n📁 Local output location: {output_base_path}")
    print("\nFiles created:")
    print("   • sent_training_data_go.tsv  - Training set")
    print("   • validate_sent_go.tsv       - Validation set")
    print("   • test_sent_go.tsv           - Test set")
    
    print("\n" + "=" * 70)
    print("📝 For NLP Scholar config (after uploading to Turing):")
    print("=" * 70)
    print(f"\ntrainfpath: {turing_output_path}sent_training_data_go.tsv")
    print(f"validfpath: {turing_output_path}validate_sent_go.tsv")
    print(f"testfpath:  {turing_output_path}test_sent_go.tsv")
    print(f"modelfpath: {turing_output_path}model_sentgo")
    
    print("\n💡 Remember to:")
    print(f"   1. Change 'username' to your Turing username in the script")
    print(f"   2. Upload the files from {output_base_path} to Turing")
    print(f"   3. Create the directory on Turing: mkdir -p {turing_output_path}")

if __name__ == "__main__":
    main()
