# Syntropy-















**Outlining from the output of NLPScholar to your evaluation metrics / table/ figures. **


To support our evaluation, we begin by preparing data from two distinct sources: the GoEmotions dataset, which contains Reddit posts labeled with 27 emotions, and a journal dataset consisting of personal entries with more than 14 emotional categories. We first identify the intersection of emotions shared between the two datasets and select eight common emotion classes: anger, confusion, disgust, excitement, joy/happiness, pride, sadness, and surprise. Using this harmonized label space, we define two modeling approaches. Method α operates at the sentence level, where each paragraph is tokenized into individual sentences, models are trained using sentence-level emotional labels, and paragraph predictions are obtained by aggregating sentence-level outputs. Method β instead treats each paragraph as a single unit, training models directly on full-paragraph inputs and corresponding paragraph-level emotion labels. After training both models, we evaluate their performance under three major settings: in-distribution performance on GoEmotions and journal data, out-of-distribution generalization between the datasets, and direct comparison of method α versus method β to assess whether sentence-derived paragraph emotion estimation can match or approximate direct paragraph-level classification.

To not the 
