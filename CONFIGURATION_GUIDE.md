# Dataset Configuration Guide

## CRITICAL NOTE

**The configurable train/validation/test splits ONLY apply to the Journal dataset.**

- **GoEmotions**: Always uses its original pre-split files (train.tsv, dev.tsv, test.tsv) from the source dataset
- **Journal**: Fully configurable for zero-shot, few-shot, or full training scenarios

This design makes sense because:
- GoEmotions is a standard benchmark with established splits
- Journal is your custom dataset that you can experiment with different training scenarios

---

## Overview

Both dataset creation scripts now support flexible train/validation/test splits for different experimental scenarios (zero-shot, few-shot, full training).

** IMPORTANT: Configurable splits apply to JOURNAL dataset ONLY.**

- **GoEmotions**: Uses its original pre-split files (train.tsv, dev.tsv, test.tsv) - NOT affected by configuration
- **Journal**: Fully configurable splits based on the settings you choose

## Modified Files

1. **`create_datasets_deduplicated.py`** - Creates deduplicated one-to-one label datasets
   - GoEmotions: Uses original splits
   - Journal: Configurable splits
   
2. **`create_datasets_journal.py`** - Creates journal-specific datasets
   - Journal: Configurable splits
   
3. **`README.md`** - Updated with configuration instructions

## Configuration Options

### At the top of each script, you'll find:

```python
### At the top of each script, you'll find:

```python
# ============================================
# SPLIT PERCENTAGES - EASY TO MODIFY (JOURNAL DATASET ONLY)
# ============================================
# NOTE: These splits only apply to the JOURNAL dataset.
# GoEmotions uses its original pre-split files (train.tsv, dev.tsv, test.tsv)

# ZERO-SHOT: All journal data goes to test (currently active)
TRAIN_SIZE = 0.00
VAL_SIZE = 0.00
TEST_SIZE = 1.00

# FEW-SHOT: Uncomment for few-shot learning
# TRAIN_SIZE = 0.05  # 5% for training
# VAL_SIZE = 0.05    # 5% for validation
# TEST_SIZE = 0.90   # 90% for testing

# STANDARD: Uncomment for full training
# TRAIN_SIZE = 0.70
# VAL_SIZE = 0.15
# TEST_SIZE = 0.15
```
```

## How to Switch Between Scenarios

**Remember: These settings only affect the Journal dataset!**  
GoEmotions always uses its original train/dev/test split files.

### 1. Zero-Shot Testing (Current Default)
- Use when you want to evaluate Journal data without any training
- All Journal data goes to test set
- GoEmotions keeps its original splits
- Keep the ZERO-SHOT section uncommented

```python
TRAIN_SIZE = 0.00
VAL_SIZE = 0.00
TEST_SIZE = 1.00
```

### 2. Few-Shot Learning
- Use for few-shot experiments with Journal data (5%, 10%, etc.)
- GoEmotions keeps its original splits
- Comment out ZERO-SHOT, uncomment FEW-SHOT
- Adjust percentages as needed

```python
# TRAIN_SIZE = 0.00  # ← Comment out zero-shot
# VAL_SIZE = 0.00
# TEST_SIZE = 1.00

TRAIN_SIZE = 0.05  # ← Uncomment few-shot
VAL_SIZE = 0.05
TEST_SIZE = 0.90
```

### 3. Standard Training
- Use for full Journal model training
- GoEmotions keeps its original splits
- Comment out ZERO-SHOT, uncomment STANDARD

```python
# TRAIN_SIZE = 0.00  # ← Comment out zero-shot
# VAL_SIZE = 0.00
# TEST_SIZE = 1.00

TRAIN_SIZE = 0.70  # ← Uncomment standard
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```

### 4. Custom Splits
- Set any percentages you need
- Must sum to 1.0

```python
TRAIN_SIZE = 0.10  # 10% training
VAL_SIZE = 0.10    # 10% validation
TEST_SIZE = 0.80   # 80% testing
```

## Key Features

### GoEmotions Dataset
- **Always uses original pre-split files** from the source
- train.tsv → Training set (predefined)
- dev.tsv → Validation set (predefined)
- test.tsv → Test set (predefined)
- **Not affected by configuration settings**

### Journal Dataset
- **Fully configurable splits** based on your settings
- Supports zero-shot, few-shot, and full training scenarios

### Deduplicated One-to-One Mapping
- Each text appears exactly **once** in the dataset
- For multi-label texts, one emotion is randomly selected
- Ensures true one-to-one text-to-label mapping

### Stratified Splits
- Emotion distribution is maintained across train/val/test sets
- Prevents bias toward certain emotions in specific splits

### Handles Edge Cases
- Zero-shot mode bypasses sklearn's train_test_split entirely
- Automatic validation for edge cases (100% test, 0% train, etc.)
- Prevents sklearn errors with invalid split sizes

## Example Workflows

### Scenario 1: Zero-Shot Evaluation
```bash
# Keep default configuration in both scripts
python create_datasets_deduplicated.py
# GoEmotions → Uses original train/dev/test splits
# Journal → Creates empty train/val, full test set
```

### Scenario 2: 5% Few-Shot Experiment
```python
# Edit both scripts:
TRAIN_SIZE = 0.05
VAL_SIZE = 0.05
TEST_SIZE = 0.90
```
```bash
python create_datasets_deduplicated.py
# GoEmotions → Uses original train/dev/test splits (unchanged)
# Journal → Creates 5% train, 5% val, 90% test
```

### Scenario 3: Full Training (70/15/15)
```python
# Edit both scripts:
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```
```bash
python create_datasets_deduplicated.py
# GoEmotions → Uses original train/dev/test splits (unchanged)
# Journal → Creates 70% train, 15% val, 15% test
```

## Output Files

After running the scripts, you'll get three .tsv files for each dataset:

### GoEmotions (always from original splits):
1. **`sent_training_data_go.tsv`** - Training set (from train.tsv)
2. **`validate_sent_go.tsv`** - Validation set (from dev.tsv)
3. **`test_sent_go.tsv`** - Test set (from test.tsv)

### Journal (configurable splits):
1. **`sent_training_data_journal.tsv`** - Training set (can be empty for zero-shot)
2. **`validate_sent_journal.tsv`** - Validation set (can be empty for zero-shot)
3. **`test_sent_journal.tsv`** - Test set (contains all data for zero-shot)

## Verification

The scripts will print:
- Split configuration being used
- Number of examples in each split
- Percentage distribution
- Emotion distribution per split

Example output:
```
Split configuration: 5% train / 5% val / 90% test

Split sizes:
   Training:      50 (5.0%)
   Validation:    50 (5.0%)
   Test:         900 (90.0%)
   Total:      1,000
```

## Tips

- Always make sure percentages sum to 1.0
- **Use the same configuration in both scripts for consistency** (applies to Journal only)
- **GoEmotions splits are fixed** - they come from the original dataset files
- **Journal splits are flexible** - adjust based on your experimental needs
- For reproducibility, the random_state is fixed at 42
- Zero-shot is best for pure evaluation of Journal data without any training
- Few-shot (5-10%) is good for testing learning efficiency with limited Journal data
- Standard (70/15/15) is best for full Journal model development

## Questions?

If you need different split configurations or have questions about the setup, just modify the TRAIN_SIZE, VAL_SIZE, and TEST_SIZE values at the top of each script!
