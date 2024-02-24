import os
import soundfile as sf
import numpy as np

# Configuración de parámetros
# Carpeta donde están los archivos .ogg de entrada
input_folder = './sonidos_minecraft/wolf'
# Carpeta donde se guardarán los archivos fragmentados
output_folder = './outputs/wolf'
sample_rate = 44100  # Frecuencia de muestreo
frame_size = 32768  # Tamaño de cada fragmento
overlap = 0.5  # Porcentaje de solapamiento

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos de entrada
input_files = [f for f in os.listdir(input_folder) if f.endswith('.ogg')]

# Procesar cada archivo de entrada
for input_file in input_files:
    # Cargar el archivo de entrada
    data, sr = sf.read(os.path.join(input_folder, input_file))
    if sr != sample_rate:
        print(
            f'Error: la frecuencia de muestreo de "{input_file}" no es {sample_rate} Hz')
        continue

    # Calcular el número de fragmentos a generar
    hop_size = int(frame_size * (1 - overlap))
    num_frames = int(np.ceil(len(data) / hop_size))

    # Fragmentar el archivo de entrada y guardar los fragmentos
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end > len(data):
            # Añadir ceros al final si el último fragmento es más corto
            padding = end - len(data)
            fragment = np.concatenate((data[start:], np.zeros(padding)))
        else:
            fragment = data[start:end]

        # Guardar el fragmento en un archivo de salida
        output_file = f'{os.path.splitext(input_file)[0]}{i:03}.wav'
        sf.write(os.path.join(output_folder, output_file),
                 fragment, sample_rate)
