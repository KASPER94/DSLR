import pandas as pd

def checker(pred_file, truth_file):
    try:
        pred_df = pd.read_csv(pred_file)
        truth_df = pd.read_csv(truth_file)
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return
    
    pred_df = pred_df.sort_values("Index").reset_index(drop=True)
    truth_df = truth_df.sort_values("Index").reset_index(drop=True)

    if len(pred_df) != len(truth_df):
        return
    
    correct = (pred_df["Hogwarts House"] == truth_df["Hogwarts House"]).sum()
    total = len(truth_df)
    accuracy = correct / total * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct} / {total} corrects)")