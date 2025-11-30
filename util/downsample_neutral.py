import pandas as pd
import os

"""
This script downsamples 'neutral' emotion examples to reduce class imbalance.
Strategy: Keep neutral examples at slightly more than the second most frequent emotion.
"""

def downsample_neutral(input_tsv_path, output_tsv_path, multiplier=1.2):
    """
    Downsample neutral emotion to be slightly more than the second largest class.
    
    Args:
        input_tsv_path: Path to input TSV file with columns [textid, text, label/target, source]
        output_tsv_path: Path to save downsampled TSV file
        multiplier: How much larger neutral should be than 2nd largest (default: 1.2 = 20% more)
    """
    
    print("="*70)
    print("DOWNSAMPLING NEUTRAL EMOTION")
    print("="*70)
    
    # Read TSV
    df = pd.read_csv(input_tsv_path, sep='\t')
    
    # Determine label column name (could be 'label' or 'target')
    label_col = 'label' if 'label' in df.columns else 'target'
    
    print(f"\n📂 Loaded {len(df)} examples")
    
    # Get emotion distribution
    emotion_counts = df[label_col].value_counts()
    
    print("\n📊 Original emotion distribution:")
    for emotion, count in emotion_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {emotion:15s}: {count:5,} ({pct:5.2f}%)")
    
    # Check if neutral exists
    if 'neutral' not in emotion_counts.index:
        print("\n⚠️  No 'neutral' emotion found in dataset. Returning original dataset.")
        df.to_csv(output_tsv_path, sep='\t', index=False)
        return df
    
    # Get counts
    neutral_count = emotion_counts['neutral']
    non_neutral_counts = emotion_counts[emotion_counts.index != 'neutral']
    
    if len(non_neutral_counts) == 0:
        print("\n⚠️  Dataset only contains neutral. No downsampling needed.")
        df.to_csv(output_tsv_path, sep='\t', index=False)
        return df
    
    # Find second largest emotion (largest non-neutral)
    second_largest = non_neutral_counts.iloc[0]
    second_largest_emotion = non_neutral_counts.index[0]
    
    print(f"\n🎯 Target strategy:")
    print(f"   Second largest emotion: {second_largest_emotion} ({second_largest:,} examples)")
    print(f"   Current neutral count: {neutral_count:,}")
    
    # Calculate target neutral count (slightly more than second largest)
    target_neutral = int(second_largest * multiplier)
    
    print(f"   Target neutral count: {target_neutral:,} ({multiplier}x second largest)")
    
    if neutral_count <= target_neutral:
        print(f"\n✅ Neutral already at or below target. No downsampling needed.")
        df.to_csv(output_tsv_path, sep='\t', index=False)
        return df
    
    # Downsample neutral
    neutral_df = df[df[label_col] == 'neutral']
    non_neutral_df = df[df[label_col] != 'neutral']
    
    # Randomly sample target_neutral examples from neutral
    neutral_downsampled = neutral_df.sample(n=target_neutral, random_state=42)
    
    # Combine
    result_df = pd.concat([non_neutral_df, neutral_downsampled], ignore_index=True)
    
    # Shuffle to mix emotions
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset textid
    result_df['textid'] = range(len(result_df))
    
    # Save
    result_df.to_csv(output_tsv_path, sep='\t', index=False)
    
    print(f"\n📊 Final emotion distribution:")
    final_counts = result_df[label_col].value_counts()
    for emotion, count in final_counts.items():
        pct = (count / len(result_df)) * 100
        print(f"   {emotion:15s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"\n✅ Removed {neutral_count - target_neutral:,} neutral examples")
    print(f"📉 Dataset reduced from {len(df):,} to {len(result_df):,} examples")
    print(f"💾 Saved to: {output_tsv_path}")
    
    return result_df


def process_all_splits(input_dir, output_dir, multiplier=1.2):
    """
    Process all train/val/test splits in a directory.
    
    Args:
        input_dir: Directory containing TSV files
        output_dir: Directory to save downsampled files
        multiplier: How much larger neutral should be than 2nd largest
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to process (GoEmotions only)
    files = [
        'sent_training_data_go.tsv',
        'validate_sent_go.tsv',
        'test_sent_go.tsv'
    ]
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(input_path):
            print(f"\n{'='*70}")
            print(f"Processing: {filename}")
            print(f"{'='*70}")
            downsample_neutral(input_path, output_path, multiplier)
        else:
            print(f"\n⚠️  Skipping {filename} (not found)")


def main():
    """
    Main function to downsample neutral emotion in datasets.
    """
    
    print("="*70)
    print("NEUTRAL EMOTION DOWNSAMPLING SCRIPT")
    print("="*70)
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Input directory (deduplicated datasets)
    input_dir = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_deduplicated/'
    
    # Output directory (downsampled datasets)
    output_dir = '/Users/davidreyes/Documents/Syntropy/data/go_emotions_balanced/'
    
    # Multiplier: How much larger should neutral be than 2nd largest?
    # 1.0 = same as 2nd largest
    # 1.2 = 20% more than 2nd largest (recommended)
    # 1.5 = 50% more than 2nd largest
    NEUTRAL_MULTIPLIER = 1.2
    
    # ============================================
    
    print(f"\n📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Neutral multiplier: {NEUTRAL_MULTIPLIER}x second largest emotion")
    
    # Process all files
    process_all_splits(input_dir, output_dir, NEUTRAL_MULTIPLIER)
    
    print("\n" + "="*70)
    print("✅ NEUTRAL DOWNSAMPLING COMPLETE!")
    print("="*70)
    
    print("\n💡 Tips:")
    print("   • Adjust NEUTRAL_MULTIPLIER to control neutral proportion")
    print("   • 1.0 = neutral same as 2nd largest")
    print("   • 1.2 = neutral 20% more (good balance)")
    print("   • 1.5 = neutral 50% more (still reduces imbalance)")


if __name__ == "__main__":
    main()
