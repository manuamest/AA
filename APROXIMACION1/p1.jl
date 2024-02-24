import Pkg;
Pkg.add("DelimitedFiles");
Pkg.add("Statistics");
using Statistics
using DelimitedFiles
dataset = readdlm("iris.data", ',');
inputs = dataset[:, 1:4];
#inputs:
inputs = dataset[:, 1:4];
#outputs deseadas:
targets = dataset[:, 5];
#vamos a convertir los datos al tipo correcto:  

inputs = convert(Array{Float32,2}, inputs);

#Vimos arbol de tipos de datos y como convertir usando funciones,
#tambien vimos el <: que sirve para ver de que tipo de dato es un dato


#Normalmente las entradas serán de tipo Float32 y las salidas dependiendo
#del problema serán de un tipo u de otro, por ejemplo una salida con 2 opciones
#será de tipo bool


#Como vimos las filas de la entrada y salida deben ser las mismas, así que vamos
#a comprobarlo

@assert (size(inputs, 1) == size(targets, 1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"


#Vamos a crear una funcion que reciba como parámetros un vector con las salidas,
#y lo vamos a codificar usando unique para ver el numero de elementos diferentes
#y ver que tipo de codificacion usamos a usar
#M=Matrix{Int64}([1 2 3 4 5; 3 2 3 4 5])
#size(unique(M))
function output_conversion(M)
    m_unique = unique(M) #cogemos la matriz de clases
    S = length(m_unique) #miramos la longitud de la matriz
    @assert (S > 1) "El número de elementos es demasiado pequeño"
    if S == 2 #miramos si el tamaño de las clases es 2
        resultado = Array{Bool,2}(undef, size(M, 1), 1) #generamos un array
        resultado[:, 1] .= (M .== m_unique[1]) #esto viene siendo equivalente a resultado = (M .== m_unique[1])
        #la operacion M .== m_unique[1] devolvera una matriz de booleanos con true donde M[i] es igual al elemento de m_unique[1]
        return resultado
    elseif S > 2 #miramos si el tamaño de las clases es mayor que 2
        resultado = Array{Bool,2}(undef, size(M, 1), S)
        for n = 1:S
            resultado[:, n] .= (M .== m_unique[n])
        end
        return resultado
    end
end

targets = output_conversion(targets);

#Cuando ponemos res[:,x] .= M, siendo M un vector con las mismas filas que res, lo que hará será asignar el resultado
#de la oeperacion sobre M en esa columna x de res y en todas sus filas
#Despues de tener la función anterior solo tenemos que calcular algunos datos de la entrada
#y normalizarla
#con el ; evitamos que el interprete saque la salida
#sacamos maximo, minimo, media y desviacion típica:
max_input = maximum(inputs, dims=1);
min_input = minimum(inputs, dims=1);
mean_input = mean(inputs, dims=1);
std_input = std(inputs, dims=1);
#ahora normalizamos la matriz por max min
#en la columna que max y min son igual a 0, restar
function normalizarmaxmin(M, max, min)
    if (max == min)
        return 0
    else
        return ((M - min) / (max - min))
    end
end
inputs = normalizarmaxmin.(inputs, max_input, min_input)

#ahora normalizamos la matriz por media
difvmean = inputs .- mean_input
inputsmeanstd = difvmean ./ std_input
function normalizarmediastd(M, media, desvi)
    if (desvi == 0)
        return 0
    else
        return ((M - media) / (desvi))
    end
end
inputs = normalizarmediastd.(inputs, mean_input, std_input)
< >
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)
    #Vamos a crear la RNA
    ann = buildClassANN(size(trainingDataset[1], 2), topology, size(trainingDataset[2], 2), transferFunctions)
    #Definimos la funcion de loss
    loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)
    #Creamos los vectores a devolver
    trainingL = Float64[]
    trainingA = Float64[]
    validationL = Float64[]
    validationA = Float64[]
    testL = Float64[]
    testA = Float64[]
    ciclo = 0
    #como vamos a realizar lo mismo varias veces creamos la siguiente funcion:
    function calcularParametros()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasarlas matrices traspuestas (cada patron en una columna)
        trainL = loss(trainingDataset[1]', trainingDataset[2]')
        valL = 0.0
        testL = 0.0
        validationAcc = 1.0
        testAcc = 1.0
        trainingOutputs = ann(trainingDataset[1]')
        trainingAcc = accuracy(trainingOutputs, trainingDataset[2]')
        if (length(validationDataset[1]) > 0 && length(validationDataset[2]) > 0)
            valL = loss(validationDataset[1]', validationDataset[2]')
            validationOutputs = ann(validationDataset[1]')
            validationAcc = accuracy(validationOutputs, validationDataset[2]')
        end
        if (length(testDataset[1]) > 0 && length(testDataset[2]) > 0)
            testL = loss(testDataset[1]', testDataset[2]')
            testOutputs = ann(testDataset[1]')
            testAcc = accuracy(testOutputs, testDataset[2]')
        end
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainL, ",
            accuracy: ", 100 * trainingAcc, " % - Validation loss: ", valL, ",
              accuracy: ", 100 * validationAcc, " % - Test loss: ", testL, ", accuracy: ",
                100 * testAcc, " %")
        end
        return (trainL, trainingAcc, valL, validationAcc, testL, testAcc)
    end
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calcularParametros()

    push!(trainingL, trainingLoss)
    push!(trainingA, trainingAccuracy)
    if (length(validationDataset[1]) > 0 && length(validationDataset[2]) > 0)
        push!(validationL, validationLoss)
        push!(validationA, validationAccuracy)
    end
    if (length(testDataset[1]) > 0 && length(testDataset[2]) > 0)
        push!(testL, testLoss)
        push!(testA, testAccuracy)
    end
    bestANN = deepcopy(ann)
    while (ciclo < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
        Flux.train!(loss, params(ann), [(inputs1, targets1)], ADAM(learningRate))
        ciclo += 1
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calcularParametros()
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)
        if (length(validationDataset[1]) > 0 && length(validationDataset[2]) > 0)
            push!(validationLosses, validationLoss)
            push!(validationAccuracies, validationAccuracy)
        end
        if (length(testDataset[1]) > 0 && length(testDataset[2]) > 0)
            push!(testLosses, testLoss)
            push!(testAccuracies, testAccuracy)
        end
        if (length(validationDataset[1]) > 0 && length(validationDataset[2]) > 0)
            if (validationLoss < bestValidationLoss)
                bestValidationLoss = validationLoss
                numEpochsValidation = 0
                bestANN = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
        else
            bestANN = ann
        end
    end
    return (bestANN, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies)

end