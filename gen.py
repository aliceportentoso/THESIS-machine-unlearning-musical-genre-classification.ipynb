from IPython.core.display_functions import display
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import numpy as np
from IPython.display import Audio
import os
from config import *
import simpleaudio as sa

model_name = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name)

prompt = "lo-fi track with a relaxing melody and soft drums"

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
)

audio_values = model.generate(
    **inputs,
    max_new_tokens=1024,   # ~50 secondi di audio (aumenta o diminuisci)
)

audio_numpy = audio_values[0].cpu().numpy().squeeze().astype("float32")
audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))  # normalizzazione

sampling_rate = model.config.audio_encoder.sampling_rate

# Preview audio direttamente in notebook
display(Audio(audio_numpy, rate=sampling_rate))

os.makedirs("gen_output", exist_ok=True)
output_path = f"gen_output/output_{Config.timestamp}.wav"

sf.write(output_path, audio_numpy, sampling_rate)

print(f"Audio salvato in: {output_path}")

wave_obj = sa.WaveObject.from_wave_file("gen_output/output.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
