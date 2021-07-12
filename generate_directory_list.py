import numpy as np
import pandas as pd
import glob
import os
  
import librosa
import soundfile as sf


def check_utt_length(file_path):
    main_path = file_path.rsplit("/", 2)[0]
    file_name = file_path.split("/")[-1].split(".")[0]

    wav_location = f"{main_path}/wav_arrayMic/{file_name}.wav"
    
    y, sr = librosa.load(wav_location, 16000)
    duration = librosa.get_duration(y=y, sr=sr)

    return duration


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
    transcripts = []
    directories = []
    utt_type = []
    file_duration = []

    print("Analyzing files...")

    for og_file in all_files:
        file_ = og_file.replace("\\", "/")
        f_ = open(file_, "r")

        general_id = file_.split("/")[1]
        general_ids.append(general_id)
        transcript_prompt = f_.read()
        transcript_prompt = transcript_prompt.strip("\n")
        transcripts.append(transcript_prompt)

        directories.append(file_)
        utt_type.append(define_utt_type(transcript_prompt))

        
        try:
            duration = check_utt_length(file_)
            file_duration.append(duration)
        
        except FileNotFoundError as e:
            file_duration.append(np.NaN)
        


    df = pd.DataFrame({
        "general_ids": general_ids,
        "directory": directories,
        "transcripts": transcripts,
        "utt_type": utt_type,
        "duration": file_duration
    })

    df.to_csv(f"transcripts.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    check_transcripts("./*/*0*/Session*/prompts/*.txt")
