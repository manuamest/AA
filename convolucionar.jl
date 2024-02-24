

# Función para generar un array con las señales de frecuencia de un archivo de audio
function generar_array_audios(audio_file)
    audio, Fs = wavread(audio_file)
    senalFrecuencia = abs.(fft(audio))
    return senalFrecuencia
end
# Ahora array_senales es un array de arrays, donde cada subarray contiene la señal de frecuencia de un archivo de audio
zombie = ["zombie", "zombiepig", "zombie_villlager", "drowned"] #ponemos los que son zombies
nozombie = ["oveja", "spider", "vaca", "villager", "skeleton", "wolf"] #ponemos los que no son zombies
# Carpeta de entrada
#input_folder = "u"
input_folder = "/Users/fiopans1/git/class-repositories/pracicas_AA/outputs/"
# Inicializar un array vacío para almacenar las señales de frecuencia de cada archivo de audio

array_senales = Array{Float32,4}(undef, size(generar_array_audios("/Users/fiopans1/git/class-repositories/pracicas_AA/outputs/oveja/say1000.wav"), 1), 1, 1, 185)  # inicializar el array sin conocer el tamaño de cada 
array_sol = Array{Int32,1}(undef, 185)
i = 0
# Iterar sobre las carpetas de la carpeta de entrada
for folder in readdir(input_folder)
    # Obtener lista de archivos OGG en la carpeta actual
    audio_files = glob("*.wav", joinpath(input_folder, folder))
    # Iterar sobre los archivos de audio y almacenar sus señales de frecuencia en el array
    for audio_file in audio_files
        global i
        i += 1
        array_senales[:, 1, 1, i] .= generar_array_audios(audio_file)
        if folder in zombie
            array_sol[i] = 1
        else
            array_sol[i] = 0
        end
    end
end

print(array_sol)