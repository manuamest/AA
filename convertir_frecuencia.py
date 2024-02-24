import os
from pydub import AudioSegment

# Carpeta de entrada con los archivos de audio
carpeta_entrada = "./sonidos_minecraft/drowned"
# Carpeta de salida para guardar los archivos de audio con frecuencia de 44100 Hz
carpeta_salida = "./sonidos_minecraft/drowned2"

# Obtener todos los archivos de audio en la carpeta de entrada
archivos_audio = [f for f in os.listdir(carpeta_entrada) if f.endswith((".mp3", ".wav", ".ogg"))]

# Iterar sobre los archivos de audio
for archivo_audio in archivos_audio:
    # Cargar el archivo de audio con PyDub
    ruta_archivo_audio = os.path.join(carpeta_entrada, archivo_audio)
    audio = AudioSegment.from_file(ruta_archivo_audio)
    
    # Convertir la frecuencia de muestreo a 44100 Hz
    audio = audio.set_frame_rate(44100)
    
    # Guardar el archivo de audio en la carpeta de salida con la misma extensión de archivo
    nuevo_nombre_archivo = archivo_audio.split(".")[0] + "_44100Hz." + archivo_audio.split(".")[1]
    ruta_nuevo_archivo = os.path.join(carpeta_salida, nuevo_nombre_archivo)
    audio.export(ruta_nuevo_archivo, format=archivo_audio.split(".")[1])
    
    print(f"Archivo de audio '{archivo_audio}' transformado a 44100 Hz y guardado como '{nuevo_nombre_archivo}' en la carpeta de salida.")
