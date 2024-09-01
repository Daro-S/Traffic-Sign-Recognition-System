import pandas as pd
from gtts import gTTS
import os

csv_file = "Labels.csv"  
data = pd.read_csv(csv_file)

output_dir = "descriptions_mp3"
os.makedirs(output_dir, exist_ok=True)

for index, row in data.iterrows():
    class_id = row['ClassId']
    description = row['Description']
    
    tts = gTTS(text=description, lang='en')  
    
    mp3_filename = f"{output_dir}/{class_id}.mp3"
    tts.save(mp3_filename)
    
    print(f"Saved {mp3_filename}")

print("All descriptions have been converted and saved as MP3 files.")
