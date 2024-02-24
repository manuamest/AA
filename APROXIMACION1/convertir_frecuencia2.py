from pydub import AudioSegment
import os

# Carpeta donde se encuentran los archivos de audio
folder_path = "ruta/a/carpeta"

# Frecuencia objetivo (44100 Hz)
target_freq = 44100

# Iterar a través de todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        # Cargar el archivo de audio
        sound = AudioSegment.from_file(os.path.join(folder_path, filename))

        # Obtener la frecuencia actual del archivo
        current_freq = sound.frame_rate

        # Verificar si la frecuencia del archivo es diferente a la frecuencia objetivo
        if current_freq != target_freq:
            # Cambiar la frecuencia del archivo
            sound = sound.set_frame_rate(target_freq)

            # Guardar el archivo modificado en la misma ubicación y con el mismo nombre
            sound.export(os.path.join(folder_path, filename), format="mp3")
            print(
                f"El archivo {filename} ha sido modificado a una frecuencia de {target_freq} Hz")
        else:
            print(
                f"El archivo {filename} ya tiene una frecuencia de {target_freq} Hz")
