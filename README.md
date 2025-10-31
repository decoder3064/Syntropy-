# Syntropy-


### Outlining a sequence of steps required to go from the dataset to the sample tsv files. 

#### Step 1:



















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

