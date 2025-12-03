import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def random_baseline(train_df, test_df, label_col, save_path=None, random_state=None):
    """
    Generates random predictions for test_df based on label distribution in train_df using pandas.sample.

    Args:
        train_df (pd.DataFrame): training data containing labels
        test_df (pd.DataFrame): test data to predict on
        label_col (str): name of the label column
        save_path (str): optional path to save predictions
        random_state (int): optional random seed for reproducibility

    Returns:
        pd.Series: predicted labels for test_df
    """


    # Sample labels for test_df based on distribution
    preds = train_df[label_col].sample(
        n=len(test_df),
        replace=True,
        random_state=random_state
    )

    # Optionally save
    if save_path:
        preds.to_csv(save_path, index=False)

    return preds


def evaluate_random_baseline(test_df, predictions, label_col):

    true_labels = test_df[label_col]

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "confusion_matrix": confusion_matrix(true_labels, predictions),
        "classification_report": classification_report(true_labels, predictions, output_dict=True)
    }


def main():
    
    test_df = pd.read_csv()
    train_df = pd.read_csv()
    
    preds = random_baseline(train_df, test_df, label_col="sentiment", random_state=1313)
    metrics = evaluate_random_baseline(test_df, preds, label_col="sentiment")

    print("Accuracy:", metrics["accuracy"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    
if __name__ == "__main__":
    main()