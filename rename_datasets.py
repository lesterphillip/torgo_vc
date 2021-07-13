import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import librosa
import soundfile as sf


def open_and_save_wav(file_path, new_id):
    y, sr = librosa.load(file_path, 16000)
    general_id = file_path.split("/")[1]
    new_path = f"output/{general_id}/{new_id}.wav"
    sf.write(new_path, y, sr)


def process_csv_file(df_dys, df_nondys, gender):

    df_res = pd.merge(df_dys, df_nondys, on="transcripts")
    df_res = df_res.loc[df_res["transcripts"] != "[relax your mouth in its normal position]"]
    df_res = df_res.drop_duplicates(subset="directory_x")
    df_res = df_res.reset_index(drop=True)

    print(f"Dysarthric duration: {df_res['duration_x'].sum()}")
    print(f"Non-dysarthric duration: {df_res['duration_y'].sum()}")

    df_res.to_csv(f"paired_transcripts_{gender}.csv", index=False)

    for index, row in tqdm(df_res.iterrows(), total=len(df_res)):
        wav_f = row["directory_x"]
        wav_fc = row["directory_y"]

        if ".wav" not in str(wav_f) or ".wav" not in str(wav_fc):
            continue

        new_id = str(index).zfill(4)
        open_and_save_wav(wav_f, new_id)
        open_and_save_wav(wav_fc, new_id)

    df_res.to_csv("total_summary.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv("transcripts.csv")
    df_dysF = df.loc[df["general_ids"] == "F"]
    df_nondysF = df.loc[df["general_ids"] == "FC"]

    df_dysM = df.loc[df["general_ids"] == "M"]
    df_nondysM = df.loc[df["general_ids"] == "MC"]

    process_csv_file(df_dysF, df_nondysF, "F")
    process_csv_file(df_dysM, df_nondysM, "M")