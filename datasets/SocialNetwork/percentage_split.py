import pandas as pd
import os
from sklearn.model_selection import train_test_split

percentages = [80, 60, 40, 20]   #2,4,8,16, 5, 10, 20, 30, 40, 50, 60
folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
label_key = "label"     # "label"
id_key = "account_id"   # "account_id"
setting = "r"    # ["r", "rs", "s"]
BASE_DIR = "datasets/SocialNetwork/user_classification"

for percentage in percentages:
    print(f"PERCENTAGE: {percentage}\n")
    for fold in folds:
        dst_dir = os.path.join(BASE_DIR, f"percented_train_{setting}", f"{percentage}_perc", "cross_validation",
                               f"fold_{fold}")
        os.makedirs(dst_dir, exist_ok=True)
        if setting == "r":
            train_dir = os.path.join(BASE_DIR, "cross_validation", f"fold_{fold}")
            train_df = pd.read_csv(os.path.join(train_dir, "train.tsv"), sep="\t")
            real_df = train_df[train_df["account_id"].astype(str).str.startswith("r")]
            reduced_real, _ = train_test_split(
                real_df,
                train_size=percentage / 90,
                stratify=real_df[label_key],
                random_state=42  # for reproducibility
            )
            percented_train_df = reduced_real
        elif setting == "rs":
            synthetic_df = pd.read_csv("datasets/SocialNetwork/raw/synthetic_posts.tsv", sep="\t")
            synthetic_df = synthetic_df.drop(columns=[c for c in synthetic_df.columns if c not in ["account_id", "id", "content", "exact_level_found"]])

            train_dir = os.path.join(BASE_DIR, "percented_train_r", f"{percentage}_perc", "cross_validation", f"fold_{fold}")
            train_df = pd.read_csv(os.path.join(train_dir, "train.tsv"), sep="\t")
            percented_train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        else:
            raise ValueError("Invalid setting: must either be 'r' or 'rs'")

        test_df = pd.read_csv(os.path.join(train_dir, "test.tsv"), sep="\t")
        percented_train_ids = percented_train_df.drop(columns=[c for c in percented_train_df.columns if c != id_key])
        test_ids = test_df.drop(columns=[c for c in test_df.columns if c != id_key])

        percented_train_df.to_csv(os.path.join(dst_dir, "train.tsv"), sep="\t", index=False)
        percented_train_ids.to_csv(os.path.join(dst_dir, "train_ids.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(dst_dir, "test.tsv"), sep="\t", index=False)
        test_ids.to_csv(os.path.join(dst_dir, "test_ids.tsv"), sep="\t", index=False)

        print(f"ORIGINAL LENGTH: {len(train_df)}")
        print(f"CURRENT LENGTH: {len(percented_train_df)}")