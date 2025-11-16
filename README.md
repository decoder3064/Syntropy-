# Syntropy-

## Dataset Split Configuration Guide

**Important: Configurable splits apply to JOURNAL dataset only.**  
GoEmotions uses its original pre-split files (train.tsv, dev.tsv, test.tsv).

Both `create_datasets_deduplicated.py` and `create_datasets_journal.py` now support easy configuration for different training scenarios with the Journal dataset. Simply edit the configuration at the top of each file.

### Quick Configuration

At the top of each script, you'll find split percentage configurations (for Journal data only):

```python
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
```

### How to Use

1. **For Zero-Shot Testing**: Leave the default (0/0/100) - all journal data goes to test set
2. **For Few-Shot Learning**: Uncomment the few-shot section and comment out zero-shot
3. **For Full Training**: Uncomment the standard section and comment out others
4. **For Custom Splits**: Uncomment custom section and set your own percentages (must sum to 1.0)

### Important Notes

- Percentages must sum to 1.0 (100%)
- **GoEmotions keeps its original train/dev/test splits** (not affected by these settings)
- **Journal dataset splits are configurable** for different experimental scenarios
- For one-to-one label mapping (deduplicated), each text appears exactly once
- Splits use stratification to maintain emotion distribution across sets
- Changes apply to Journal dataset only

---

### Outlining a sequence of steps required to go from the dataset to the sample tsv files. 

#### Steps:
1. Load datasets (Goemotions/Journal).
2. Identify shared labels(emotions) on both datasets.
5. Curate emotions and leave the non overlaping examples aside.
6. Create paragraph datasets for both Goemotions and Journal -> specify headers: text, souce(goemotions/journal),label(emotion), and type(paragraph).
7. Create setence dataset (by dividing each paragraph into sentences) for both Goemotions and Journal -> specify headers: text, souce(goemotions/journal),label(emotion), and type(sentence).
8. Manually label sentences in paragraphs for Journal dataset, and assume that all sentences in the goemotion dataset shared the same label as its pargraph.
10. Create training, validation, and testing sets splitting the data into 70% training data, 15% validation, and 15% test data.
   


### Outlining from the output of NLPScholar to your evaluation metrics / table/ figures. 
-------

To support our evaluation, we begin by preparing data from two distinct sources: the GoEmotions dataset, which contains Reddit posts labeled with 27 emotions, and a journal dataset consisting of personal entries with more than 14 emotional categories. We first identify the intersection of emotions shared between the two datasets and select eight common emotion classes: anger, confusion, disgust, excitement, joy/happiness, pride, sadness, and surprise. Using this harmonized label space, we define two modeling approaches. Method α operates at the sentence level, where each paragraph is tokenized into individual sentences, models are trained using sentence-level emotional labels, and paragraph predictions are obtained by aggregating sentence-level outputs. Method β instead treats each paragraph as a single unit, training models directly on full-paragraph inputs and corresponding paragraph-level emotion labels.



#### For accuracy performance level results:


   After training both models, we evaluate their performance under three major settings: in-distribution performance on GoEmotions and journal data, out-of-distribution generalization between the datasets, and direct comparison of method α versus method β to assess whether sentence-derived paragraph emotion estimation can match or approximate direct paragraph-level classification. Specifically:
  
  - For Paragraph labels: we can evaluate and report both method's indistribution and out distribution pargraph label against the true labels in each dataset.
  - For Sentence labels: we can evaluate and report α's accuracy against our hand labeled sentence labels.

For 2 * 2 * 2 matrix that is 2 (method,distribution,level) we can display the accuracy
  




#### For User emotion informative results:


Sentence-level emotion distribution(sentence level detailed information)

- Shows the model’s predicted emotion probabilities for each sentence.

- Provides fine-grained insight into emotional shifts within text.
- This could be shown with one or 2 sentence examples

Within-paragraph sentence prediction summary

- Aggregates sentence-level predictions inside a paragraph and show its distribution.
e.g., Happy: 1, Sad: 2, Angry: 1

- Helpful for understanding internal emotional composition, though less informative for very short journal entries


 Emotion distribuion by Emotion Category(most useful)

* Sentence-level distribution

   + Sentence level distribution, predicted emotion label frequency of  what predicted sentence emotion appearedfor a specified emotion set(or in other words paragraph label)
   + A probability distribution matrix where the categries are the emotions and for each emotion we have a probability distribtion where the x axis are all the emotion superset and y is density(we can also do a histrogram instead where y is the count)

