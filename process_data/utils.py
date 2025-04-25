from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

import os

from config import config

def load_files():

    print("\tLoading the Image Key exel file")

    # Loading the Image Key exel file
    df = pd.read_excel(config.label_path, sheet_name="Sheet1", header=1)
    df = df[['Image No', 'Case ID', 'Category']].copy()

    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a new column to store fold assignment
    df['fold'] = -1

    # Apply the split and assign folds
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df['Category'], groups=df['Case ID'])):
        df.loc[val_idx, 'fold'] = fold

    # Save the DataFrame to a CSV
    df.to_csv(r"C:\Users\vijay\Neuro_BioMark\ALS_BioMark_VJ_V2\dataset\ROI_with_folds.csv", index=False)
    config.label_path = r"C:\Users\vijay\Neuro_BioMark\ALS_BioMark_VJ_V2\dataset\ROI_with_folds.csv"

    print("\tSaved Image Key exel file with fold assignments!")


def get_train_test(fold):
    # Load CSV
    df = pd.read_csv(config.label_path)

    # Label mapping
    label_map = {
        "Control": 0,
        "Concordant": 1,
        "Discordant": 2
    }

    # Split based on fold
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]

    # Create image paths and labels
    def make_paths_and_labels(sub_df):
        image_paths = [os.path.join(config.image_directory_path, f"{int(row['Image No'])}.tif") for _, row in
                       sub_df.iterrows()]
        labels = [label_map[row['Category']] for _, row in sub_df.iterrows()]
        CaseIds = [row["Case ID"] for _, row in sub_df.iterrows()]
        return image_paths, labels, CaseIds

    train_image_paths, train_labels, train_CaseIds = make_paths_and_labels(train_df)
    val_image_paths, val_labels, val_CaseIds = make_paths_and_labels(val_df)

    if any(item in set(val_CaseIds) for item in set(train_CaseIds)):
        print("\tThere is overlap between training and testing data.")
    else:
        print("\tNo overlap between training and testing data.")

    return train_image_paths, val_image_paths, train_labels, val_labels

