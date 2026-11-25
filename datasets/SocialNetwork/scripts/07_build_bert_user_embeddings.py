import pickle
import torch
import pandas as pd

BERT_SOURCE_DIR = "D:/UNIVERSITA'/TIROCINIO/Tesi Antolino Emanuele"
LP_DIR = "datasets/SocialNetwork/user_classification/percentages_8_64"

post_embeddings = torch.load(f"{LP_DIR}/percented_train_rs/post_bert_embeddings.pt", map_location="cpu", weights_only=False)
posts = pd.read_csv(f"{BERT_SOURCE_DIR}/real_and_synthetic_posts.tsv", sep="\t", dtype=str, usecols=["account_id"])
assert post_embeddings.shape[0] == len(posts), "Tensore e file dei post non allineati"

with open(f"{LP_DIR}/percented_train_rs/users_bert_embs_dict_avg.pkl", "rb") as f:
    real_embeddings = pickle.load(f)

rows_per_user = {}
for i, account_id in enumerate(posts["account_id"].str.strip()):
    if account_id not in rows_per_user:
        rows_per_user[account_id] = []
    rows_per_user[account_id].append(i)

embeddings = {}
for account_id, vector in real_embeddings.items():
    embeddings[account_id] = vector.cpu().float()

num_synthetic = 0
num_checked = 0
for account_id, rows in rows_per_user.items():
    average = post_embeddings[rows].mean(dim=0)
    if account_id.startswith("s"):
        embeddings[account_id] = average
        num_synthetic += 1
    elif account_id in real_embeddings:
        assert torch.allclose(average, embeddings[account_id], atol=1e-4), f"Media diversa dal pkl per {account_id}"
        num_checked += 1

print(f"Reali dal pkl: {len(real_embeddings)} (verificati: {num_checked})   Sintetici calcolati: {num_synthetic}")
torch.save(embeddings, f"{LP_DIR}/users_bert_embs_rs.pt")
print("Salvato users_bert_embs_rs.pt")