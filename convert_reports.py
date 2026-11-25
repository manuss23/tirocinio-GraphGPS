import os
import glob
import pandas as pd

RESULTS_DIR = "results"


def parse_rows(lines):
    rows = {}
    accuracy = None

    for line in lines:
        parts = line.split()
        if len(parts) == 0 or parts[0] == "precision":
            continue

        if parts[0] == "accuracy":
            accuracy = [None, None, float(parts[1]), int(parts[2])]
        else:
            name = " ".join(parts[:-4])
            rows[name] = [float(parts[-4]), float(parts[-3]), float(parts[-2]), int(parts[-1])]

    if accuracy is not None:
        rows["accuracy"] = accuracy

    return pd.DataFrame.from_dict(rows, orient="index", columns=["precision", "recall", "f1-score", "support"])


for txt_path in glob.glob(os.path.join(RESULTS_DIR, "*", "*", "*", "test", "classification_report.txt")):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()

    best_epoch = None
    mae = None
    rmse = None
    classi_lines = []
    binario_lines = []
    current = classi_lines

    for line in lines:
        if line.startswith("Epoca migliore"):
            best_epoch = int(line.split(";")[1])
        elif line.startswith("MAE:"):
            mae = float(line.split(":")[1])
        elif line.startswith("RMSE:"):
            rmse = float(line.split(":")[1])
            current = binario_lines
        else:
            current.append(line)

    errori = pd.DataFrame({"Valore": [mae, rmse]}, index=["MAE", "RMSE"])
    if best_epoch is not None:
        errori.loc["Epoca migliore"] = best_epoch

    xlsx_path = txt_path.replace(".txt", ".xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        parse_rows(classi_lines).to_excel(writer, sheet_name="Classi")
        parse_rows(binario_lines).to_excel(writer, sheet_name="Binario")
        errori.to_excel(writer, sheet_name="Errori")

    print(f"Convertito {xlsx_path}")