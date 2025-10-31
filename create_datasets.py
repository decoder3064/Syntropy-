#CREATED THIS SCRIPT USING CLAUDE WITH PROMPT ENGINEERING 



import pandas as pd
import csv
import random
from typing import List, Dict, Tuple
import os

# File paths
GOEMOTIONS_PATH = '/Users/davidreyes/Documents/Syntropy/GoEmotions/data/full_dataset/goemotions_1.csv'
JOURNAL_PATH = '/Users/davidreyes/Documents/Syntropy/Journal.csv'

def load_goemotions(filepath: str) -> pd.DataFrame:
    """Load GoEmotions data from CSV"""
    df = pd.read_csv(filepath)
    print(f"GoEmotions columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    return df

def load_journal(filepath: str) -> pd.DataFrame:
    """Load Journal data from CSV"""
    df = pd.read_csv(filepath)
    print(f"Journal columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    return df

def identify_emotion_overlap(ge_df: pd.DataFrame, j_df: pd.DataFrame) -> List[str]:
    """Find overlapping emotions between datasets"""
    # GoEmotions emotion columns (excluding metadata columns)
    ge_emotions = [col for col in ge_df.columns if col not in ['text', 'id', 'author', 'subreddit', 
                                                                'link_id', 'parent_id', 'created_utc', 
                                                                'rater_id', 'example_very_unclear']]
    
    # Journal emotion columns - looking for emotion-related columns
    j_emotion_cols = [col for col in j_df.columns if 'Answer' in col and any(
        emotion in col.lower() for emotion in ['happy', 'sad', 'angry', 'anxious', 'excited',
                                               'proud', 'confused', 'disgusted', 'surprised',
                                               'afraid', 'frustrated', 'jealous', 'nostalgic',
                                               'ashamed', 'awkward', 'bored', 'calm', 'satisfied']
    )]
    
    print(f"\nGoEmotions emotions found: {ge_emotions}")
    print(f"Journal emotion columns found: {j_emotion_cols[:10]}...")  # First 10
    
    # Map Journal emotions to GoEmotions format
    emotion_mapping = {
        'happy': 'joy',
        'sad': 'sadness', 
        'angry': 'anger',
        'anxious': 'nervousness',
        'excited': 'excitement',
        'proud': 'pride',
        'confused': 'confusion',
        'disgusted': 'disgust',
        'surprised': 'surprise',
        'afraid': 'fear',
        'frustrated': 'annoyance',
        'satisfied': 'approval'
    }
    
    # Find actual overlap
    overlapping = []
    for j_emotion in emotion_mapping.keys():
        ge_emotion = emotion_mapping[j_emotion]
        if ge_emotion in ge_emotions:
            # Check if journal has this emotion column
            j_col_name = f'Answer.f1.{j_emotion}.raw' if f'Answer.f1.{j_emotion}.raw' in j_df.columns else None
            if not j_col_name:
                # Try alternate naming
                for col in j_df.columns:
                    if j_emotion in col.lower():
                        j_col_name = col
                        break
            if j_col_name:
                overlapping.append((j_emotion, ge_emotion, j_col_name))
    
    return overlapping

def process_goemotions_to_tsv(df: pd.DataFrame, selected_emotions: List[str], output_path: str, max_samples: int = 100):
    """Convert GoEmotions data to NLPScholar format"""
    processed_data = []
    
    for _, row in df.iterrows():
        text = row['text']
        # Find which emotion(s) are labeled (handle multi-label by taking first)
        for emotion in selected_emotions:
            if emotion in row and row[emotion] == 1:
                processed_data.append({
                    'text': text,
                    'label': emotion,
                    'source': 'goemotions',
                    'type': 'sentence'
                })
                break
        
        if len(processed_data) >= max_samples:
            break
    
    # Save to TSV
    result_df = pd.DataFrame(processed_data)
    result_df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    return result_df

def process_journal_to_tsv(df: pd.DataFrame, emotion_mappings: List[Tuple], output_path: str, max_samples: int = 100):
    """Convert Journal data to NLPScholar format"""
    processed_data = []
    
    for _, row in df.iterrows():
        text = row['Answer']  # Journal uses 'Answer' column for text
        
        # Skip if text is NaN or empty
        if pd.isna(text) or text == '':
            continue
            
        # Find which emotion is labeled
        for j_emotion, ge_emotion, j_col_name in emotion_mappings:
            if j_col_name and j_col_name in row:
                try:
                    if row[j_col_name] == True or row[j_col_name] == 'TRUE' or row[j_col_name] == 1:
                        processed_data.append({
                            'text': text,
                            'label': ge_emotion,  # Use GoEmotions naming
                            'source': 'journal',
                            'type': 'paragraph'
                        })
                        break
                except:
                    continue
        
        if len(processed_data) >= max_samples:
            break
    
    # Save to TSV
    result_df = pd.DataFrame(processed_data)
    result_df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    return result_df

def create_balanced_combined_dataset(ge_df: pd.DataFrame, j_df: pd.DataFrame, 
                                    selected_emotions: List[str], output_path: str,
                                    samples_per_emotion: int = 10):
    """Create a balanced dataset combining both sources"""
    combined = []
    
    for emotion in selected_emotions:
        # Get samples from each dataset for this emotion
        ge_samples = ge_df[ge_df['label'] == emotion].head(samples_per_emotion // 2)
        j_samples = j_df[j_df['label'] == emotion].head(samples_per_emotion // 2)
        
        # Add to combined list
        for _, row in ge_samples.iterrows():
            combined.append(row.to_dict())
        for _, row in j_samples.iterrows():
            combined.append(row.to_dict())
    
    # Shuffle the combined dataset
    random.shuffle(combined)
    
    # Save to TSV
    df = pd.DataFrame(combined)
    
    # Keep only text and label columns for NLPScholar
    df_nlp = df[['text', 'label']]
    df_nlp.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    
    # Also save version with metadata for analysis
    df.to_csv(output_path.replace('.tsv', '_metadata.tsv'), sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split dataset into train/val/test sets"""
    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]
    
    # Save splits
    train_df[['text', 'label']].to_csv('data/train.tsv', sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    val_df[['text', 'label']].to_csv('data/val.tsv', sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    test_df[['text', 'label']].to_csv('data/test.tsv', sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nDataset splits created:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")  
    print(f"  Test: {len(test_df)} samples")
    
    # Show label distribution
    print("\nLabel distribution in training set:")
    print(train_df['label'].value_counts())
    
    return train_df, val_df, test_df

def main():
    """Main function to generate all required TSV files"""
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    print("Loading datasets from specified paths...")
    print(f"GoEmotions: {GOEMOTIONS_PATH}")
    print(f"Journal: {JOURNAL_PATH}")
    
    # Load data
    ge_df = load_goemotions(GOEMOTIONS_PATH)
    j_df = load_journal(JOURNAL_PATH)
    
    print(f"\nGoEmotions shape: {ge_df.shape}")
    print(f"Journal shape: {j_df.shape}")
    
    # Find overlapping emotions
    print("\nFinding overlapping emotions...")
    emotion_mappings = identify_emotion_overlap(ge_df, j_df)
    
    # Use a subset of common emotions for this experiment
    # Based on the overlap, let's use these core emotions
    selected_emotions = ['joy', 'anger', 'sadness', 'surprise', 'disgust', 
                        'confusion', 'excitement', 'pride']
    
    print(f"\nSelected emotions for experiment: {selected_emotions}")
    
    # Process GoEmotions
    print("\nProcessing GoEmotions data...")
    ge_processed = process_goemotions_to_tsv(ge_df, selected_emotions, 
                                            'data/goemotions_processed.tsv', 
                                            max_samples=200)
    print(f"  Processed {len(ge_processed)} GoEmotions samples")
    
    # Process Journal  
    print("\nProcessing Journal data...")
    j_processed = process_journal_to_tsv(j_df, emotion_mappings,
                                        'data/journal_processed.tsv',
                                        max_samples=200)
    print(f"  Processed {len(j_processed)} Journal samples")
    
    # Create combined balanced dataset
    print("\nCreating combined balanced dataset...")
    combined_df = create_balanced_combined_dataset(ge_processed, j_processed,
                                                  selected_emotions,
                                                  'data/combined_emotion.tsv',
                                                  samples_per_emotion=20)
    print(f"  Created combined dataset with {len(combined_df)} samples")
    
    # Create train/val/test splits
    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = split_dataset(combined_df)
    
    print("\nDataset creation complete!")
    print("\nFiles created:")
    print("  - data/train.tsv (for training)")
    print("  - data/val.tsv (for validation)")
    print("  - data/test.tsv (for evaluation)")
    print("  - data/combined_emotion.tsv (full dataset)")
    print("  - data/combined_emotion_metadata.tsv (with source info)")

if __name__ == "__main__":
    main()