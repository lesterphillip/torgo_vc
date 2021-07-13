import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
  
import librosa
import soundfile as sf


def check_utt_length(file_path):
    main_path = file_path.rsplit("/", 2)[0]
    file_name = file_path.split("/")[-1].split(".")[0]

    wav_location = f"{main_path}/wav_arrayMic/{file_name}.wav"
    try:
        y, sr = librosa.load(wav_location, 16000)
    except FileNotFoundError as e:
        wav_location = f"{main_path}/wav_headMic/{file_name}.wav"
        y, sr = librosa.load(wav_location, 16000)

    duration = librosa.get_duration(y=y, sr=sr)

    return duration, wav_location


def define_utt_type(transcript_prompt):
    if " " not in transcript_prompt:
        return "word"

    elif "[" in transcript_prompt or "]" in transcript_prompt:
        return "blabber"

    else:
        return "sentence"


def check_transcripts(file_path):

    all_files = glob.glob(file_path, recursive=True)

    total_actions = 0
    total_words = 0
    total_sents = 0

    general_ids = []
    spkr_ids = []
    transcripts = []
    directories = []
    utt_type = []
    file_duration = []

    print("Analyzing files...")

    for og_file in tqdm(all_files):
        file_ = og_file.replace("\\", "/")
        f_ = open(file_, "r")

        general_id = file_.split("/")[1]
        general_ids.append(general_id)

        spkr_id = file_.split("/")[2]
        spkr_ids.append(spkr_id)

        transcript_prompt = f_.read()
        transcript_prompt = transcript_prompt.strip("\n")
        transcripts.append(transcript_prompt)

        utt_type.append(define_utt_type(transcript_prompt))

        try:
            duration, wav_location = check_utt_length(file_)
            file_duration.append(duration)
            directories.append(wav_location)
        
        except FileNotFoundError as e:
            file_duration.append(np.NaN)
            directories.append(np.NaN)
        
    df = pd.DataFrame({
        "general_ids": general_ids,
        "directory": directories,
        "transcripts": transcripts,
        "utt_type": utt_type,
        "duration": file_duration
    })

    df = df[df["directory"].notna()]
    df.to_csv(f"transcripts.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    check_transcripts("./*/*0*/Session*/prompts/*.txt")
