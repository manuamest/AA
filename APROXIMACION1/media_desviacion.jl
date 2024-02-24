#import Pkg
#Pkg.add("Plots")
#Pkg.add("PlotlyJS")
#Pkg.add("Statistics")
#Pkg.add("FFTW")

using Plots
using WAV
using Pkg
using Statistics
using FFTW


# Ruta del archivo de audio

# Lectura del archivo de audio
function imprimir_caracteristicas(ruta)
    audio, Fs = wavread(ruta)
    #pyplot()
    # Representación en el dominio del tiempo
    #graficaTiempo = plot(audio, label="", xaxis="Tiempo (s)")

    # Transformada de Fourier
    senalFrecuencia = abs.(fft(audio))

    # Los valores absolutos de la primera mitad de la señal deberian de ser iguales a los de la segunda mitad, salvo errores de redondeo
    # Esto se puede ver en la grafica:
    #graficaFrecuencia = plot(senalFrecuencia, label="", xaxis="Frecuencia (Hz)")
    #  pero ademas lo comprobamos en el codigo
    if (iseven(length(audio)))
        @assert(mean(abs.(senalFrecuencia[2:Int(length(audio) / 2)] .- senalFrecuencia[end:-1:(Int(length(audio) / 2)+2)])) < 1e-8)
        senalFrecuencia = senalFrecuencia[1:(Int(length(audio) / 2)+1)]
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((length(audio) + 1) / 2)] .- senalFrecuencia[end:-1:(Int((length(audio) - 1) / 2)+2)])) < 1e-8)
        senalFrecuencia = senalFrecuencia[1:(Int((length(audio) + 1) / 2))]
    end

    # Grafica con la primera mitad de la frecuencia:
    #graficaFrecuenciaMitad = plot(senalFrecuencia, label="", xaxis="Frecuencia (Hz)")

    # Representamos las 3 graficas juntas
    #display(plot(graficaTiempo, graficaFrecuencia, graficaFrecuenciaMitad, layout=(3, 1)))
    #sleep(30)

    # Frecuencia de muestreo del audio

    # A que muestras se corresponden las frecuencias indicadas
    # Como limite se puede tomar la mitad de la frecuencia de muestreo

    # Unas caracteristicas en esa banda de frecuencias
    m1 = 1
    m2 = Int(round(length(senalFrecuencia) / 4))
    print(mean(senalFrecuencia))
    print(",")
    print(std(senalFrecuencia))
    print(",")
    print(mean(senalFrecuencia[m1:m2]))
    print(",")
    print(std(senalFrecuencia[m1:m2]))
    println(",No_Zombie")
    #println(senalFrecuencia)

end

using Glob
input_folder = "/Users/fiopans1/git/class-repositories/pracicas_AA/outputs/cerdo"


# Obtener lista de archivos OGG en la carpeta de entrada
audio_files = glob("*.wav", input_folder)

for audio_file in audio_files
    #println(audio_file)
    imprimir_caracteristicas(audio_file)
end