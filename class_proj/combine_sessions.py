
import os
import pandas as pd

LOG_ROOT = "logs"
OUTPUT = "logs/all/labeled_dataset.csv"

all_rows = []

for session in os.listdir(LOG_ROOT):
    ses_path = os.path.join(LOG_ROOT, session, "labeled_dataset.csv")
    if os.path.isfile(ses_path) and session != "all":
        print(f"[+] Including session: {session}")
        df = pd.read_csv(ses_path)
        df["session"] = session
        all_rows.append(df)

if not all_rows:
    print("No sessions found.")
else:
    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs("logs/all", exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"[✓] Saved combined dataset to {OUTPUT} (rows={len(combined)})")
