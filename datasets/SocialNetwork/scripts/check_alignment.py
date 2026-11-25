import pickle
import torch
import pandas as pd

BERT_SOURCE_DIR = "D:/UNIVERSITA'/TIROCINIO/Tesi Antolino Emanuele"
LP_RS_DIR = "datasets/SocialNetwork/user_classification/percentages_8_64/percented_train_rs"

embeddings = torch.load(f"{LP_RS_DIR}/post_bert_embeddings.pt", map_location="cpu", weights_only=False)
posts = pd.read_csv(f"{BERT_SOURCE_DIR}/real_and_synthetic_posts.tsv", sep="\t", dtype=str, usecols=["account_id"])
print(F"Righe tensore: {embeddings.shape[0]}    Righe file post: {len(posts)}")

with open(f"{LP_RS_DIR}/users_bert_embs_dict_avg.pkl", "rb") as f:
    lp_avg = pickle.load(f)

rows_per_user = {}
for i, account_id in enumerate(posts["account_id"].str.strip()):
    if account_id not in rows_per_user:
        rows_per_user[account_id] = []
    rows_per_user[account_id].append(i)

checked = 0
equal = 0
for account_id in list(lp_avg.keys())[:500]:
    my_avg = embeddings[rows_per_user[account_id]].mean(dim=0)
    checked += 1
    if torch.allclose(my_avg, lp_avg[account_id].cpu().float(), atol=1e-4):
        equal += 1

print(f"Utenti reali controllati: {checked}, Utenti con embedding uguali: {equal}")