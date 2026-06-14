import pandas as pd

def LoadData(csv : str):
    df = pd.read_csv(csv)

    vocab = sorted(df["STEP"].unique())

    step_to_id = {
        step: idx + 1
        for idx, step in enumerate(vocab)
    }

    id_to_step = {
        idx: step
        for step, idx in step_to_id.items()
    }

    sequences = []

    for seq_id, group in df.groupby("SEQUENCE_ID"):
        seq = [
            step_to_id[s]
            for s in group["STEP"]
        ]
        sequences.append(seq)

    return sequences, vocab
