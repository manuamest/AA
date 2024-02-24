import os
import pydub
from pydub import AudioSegment


def change_sampling_rate_folder(input_folder, output_folder, new_rate):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ogg"):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            sound = AudioSegment.from_ogg(input_file)
            sound = sound.set_frame_rate(new_rate)
            sound.export(output_file, format="ogg")


input_folder = "./sonidos_minecraft/drowned"
ouput_folder = "./sonidos_minecraft/cleardrowned2"
change_sampling_rate_folder(input_folder, ouput_folder, 44100)
