import Pkg;
Pkg.add("DelimitedFiles");
Pkg.add("Statistics");
Pkg.add("Flux");
using Flux.Losses
using Statistics
using DelimitedFiles
using Random
dataset = readdlm("iris.data", ',');
inputs = dataset[:, 1:4];
#inputs:
inputs = dataset[:, 1:4];
#outputs deseadas:
targets = dataset[:, 5];
#vamos a convertir los datos al tipo correcto:  

inputs = convert(Array{Float32,2}, inputs);

#@assert (size(inputs, 1) == size(targets, 1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

#targets = output_conversion(targets);

#ahora normalizamos la matriz por media
#inputs = normalizarmediastd.(inputs, mean_input, std_input)

# ----------------------------------------------P2----------------------------------------------
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numclases = length(classes)
    @assert (numclases > 1) "El número de elementos es demasiado pequeño"
    if numclases == 2
        resultado = Array{Bool,2}(undef, size(M, 1), 1)
        resultado[:, 1] .= (M .== m_unique[1])
        return resultado
    elseif numclases > 2
        resultado = Array{Bool,2}(undef, size(M, 1), S)
        for n = 1:numclases
            resultado[:, n] .= (feature .== classes[n])
        end
        return resultado
    end
end
oneHotEncoding(feature::AbstractArray{<:Any,1}) = (classes = unique(feature); oneHotEncoding(feature, classes))
oneHotEncoding(feature::AbstractArray{Bool,1}) = feature;

#funciones que calculan valores
calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = (minimum(dataset, dims=1), maximum(dataset, dims=1))
calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = (std(dataset, dims=1), mean(dataset, dims=1))

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    min = normalizationParameters[1]
    max = normalizationParameters[2]
    dataset .-= min
    dataset ./= (max .- min)
    dataset[:, vec(min .== max)] .= 0 #aqui coges el dataset y en las columnas 
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(copy(dataset))

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    std = normalizationParameters[1]
    mean = normalizationParameters[2]
    dataset .-= mean.
    dataset ./= std.dataset[:, vec(std .== 0)] .= 0
end
normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(copy(dataset))

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) #revisar
    numColums = size(outputs, 2)
    if numColums == 1
        return (outputs .>= threshold) #convert(Array{Bool,2}, outputs.>=threshold)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputsB = falses(size(outputs))
        outputsB[indicesMaxEachInstance] .= true
        return outputsB
    end
end

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets)
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    numColumsOut = size(outputs, 2)
    numColumsTar = size(targets, 2)
    @assert (numColumsOut == numColumsTar) "El número de columnas debe ser la misma"
    if numColumsOut == 1
        return accuracy(outputs[:], targets[:])
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        accuracy1 = mean(correctClassifications)
        return accuracy1
    end
end
accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy((outputs .>= threshold), targets)
accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5) = accuracy(classifyOutputs(outputs, threshold), targets)

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int)
    ann = Chain()
    numInputsLayer = numInputs
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ))
        numInputsLayer = numOutputsLayer
    end
    if numOutputs == 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

#comentarle al profe el uso de esta función
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    init = 1
    for numOutputsLayer = topology #podriamos usar solo una variable, mirar teams
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[init]))
        numInputsLayer = numOutputsLayer
        init += 1
    end
    if numOutputs == 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    ann = buildClassANN(size(dataset[1], 2), topology, size(dataset[2], 2), transferFunctions)
    inputs1 = dataset[1]'
    targets1 = dataset[2]'
    loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)
    trainingLos = Float64[]
    trainingAcc = Float64[]
    ciclo = 1
    outputP = ann(inputs1) #caculamos salidas con la propia red sin entrenar
    vlose = loss(inputs1, targets1)
    vacc = accuracy(outputP, targets1)
    push!(trainingLos, vlose)
    push!(trainingAcc, vacc)
    while (ciclo <= maxEpochs) && (vlose > minLoss)
        Flux.train!(loss, params(ann), [(inputs1, targets1)], ADAM(learningRate))
        outputP = ann(inputs1)
        vlose = loss(inputs1, targets1)
        vacc = accuracy(outputP, targets1)
        ciclo += 1
        push!(trainingLos, vlose)
        push!(trainingAcc, vacc)
    end
    return (ann, trainingLosses, trainingAccuracies)
end
function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    trainClassANN(topology, (inputs, reshape(targets, size(targets, 1), 1)),
        transferFunctions, maxEpochs, minLoss, learningRate)

end

# ----------------------------------------------P3------------------------------------------------

function holdOut(N::Int, P::Real)
    @assert ((P >= 0.0) & (P <= 1))
    indices = randperm(N)
    ind = Int(round(N * P))
    return (indices[1:ind], indices[ind:N])
end
function holdOut(N::Int, Pval::Real, Ptest::Real)
    (trainval, test) = holdOut(N, Ptest)
    (train, val) = holdOut(length(trainval), ((N * Pval) / length(trainval))) #reajustamos porcentajes
    return (trainval(train), trainval(val), test)

end

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
    numEpochsValidation = 0
    bestValidationLoss = validationLoss
    bestANN = deepcopy(ann)
    while (ciclo < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
        Flux.train!(loss, params(ann), [(trainingDataset[1]', trainingDataset[2]')], ADAM(learningRate))
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

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),1}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),1}(undef, 0, 0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    trainClassANN(topology, (trainingDataset[1], reshape(trainingDataset[2], size(trainingDataset[2], 1), 1)),
        (validationDataset[1], reshape(validationDataset[2], size(validationDataset[2], 1), 1)),
        (testDataset[1], reshape(testDataset[2], size(testDataset[2], 1), 1)),
        transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal, showText)
end

# ---------------------------------------P4.1-----------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert (length(outputs) == length(targets))
    acc = accuracy(outputs, targets) # Precision, calcula el porcentaje de aciertos
    tasafallo = 1.0 - acc #tasa de fallo, es el porcentaje de fallos
    sensibilidad = mean(outputs[targets]) # Sensibilidad
    especifidad = mean(.!outputs[.!targets]) # Especificidad
    precision = mean(targets[outputs]) # Valor predictivo positivo
    NPV = mean(.!targets[.!outputs]) # Valor predictivo negativo
    if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
        sensibilidad = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
        especifidad = 1.0
        NPV = 1.0
    end
    #Si no hemos podido calcular alguno, estos toman el valor 0
    sensibilidad = isnan(sensibilidad) ? 0.0 : sensibilidad
    especifidad = isnan(especifidad) ? 0.0 : especifidad
    precision = isnan(precision) ? 0.0 : precision
    NPV = isnan(NPV) ? 0.0 : NPV
    #Ahora calculamos F1-SCORE
    F1 = (sensibilidad == precision == 0.0) ? 0.0 : 2 * (sensibilidad * precision) / (sensibilidad + precision) #calculamos la media harmonica
    #ahora cremoas la matriz
    confMatrix = Array{Int64,2}(undef, 2, 2)
    confMatrix[1, 1] = sum(.!targets .& .!outputs) # VN
    confMatrix[1, 2] = sum(.!targets .& outputs) # FP
    confMatrix[2, 1] = sum(targets .& .!outputs) # FN
    confMatrix[2, 2] = sum(targets .& outputs) # VP
    #devolvemos todo
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end
confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix((outputs .>= threshold), targets);


function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1}; weighted::Bool=true)
    (acc, errorRate, sensibilidad, especifidad, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
    println("La precision es:", acc)
    println("La tasa de fallo es:", errorRate)
    println("La sensibilidad es:", sensibilidad)
    println("La especifidad es:", especifidad)
    println("La VPP es:", precision)
    println("La VPV es:", VPV)
    println("El f1-score es:", F1)
    for i in 1:size(confMatrix, 1)[1]
        print(confMatrix[i, :])
        print("\n")
    end
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; weighted::Bool=true)
    (acc, errorRate, sensibilidad, especifidad, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
    println("La precision es:", acc)
    println("La tasa de fallo es:", errorRate)
    println("La sensibilidad es:", sensibilidad)
    println("La especifidad es:", especifidad)
    println("La VPP es:", precision)
    println("La VPV es:", VPV)
    println("El f1-score es:", F1)
    for i in 1:size(confMatrix, 1)[1]
        print(confMatrix[i, :])
        print("\n")
    end
end

# --------------------------------------------P4.2-----------------------------------------
# Elegir una oneVSall, la segunda es la mas adecuada

function oneVSall(inputs::Array{Float64,2}, targets::Array{Bool,2})
    numClasses = size(targets, 2)
    # Nos aseguramos de que hay mas de dos clases
    @assert(numClasses > 2)
    outputs = Array{Float64,2}(undef, size(inputs, 1), numClasses)
    for numClass in 1:numClasses
        model = fit(inputs, targets[:, [numClass]])
        outputs[:, numClass] .= model(inputs)
    end
    # Aplicamos la funcion softmax
    outputs = collect(softmax(outputs')')
    # Convertimos a matriz de valores booleanos
    outputs = classifyOutputs(outputs)
    classComparison = (targets .== outputs)
    correctClassifications = all(classComparison, dims=2)
    return mean(correctClassifications)
end;

function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numClasses = size(targets, 2)
    # Nos aseguramos de que hay mas de dos clases
    @assert(numClasses > 2)
    outputs = Array{Float32,2}(undef, numInstances, numClasses)
    for numClass in 1:numClasses
        newModel = deepcopy(model)
        fit!(newModel, inputs, targets[:, [numClass]])
        outputs[:, numClass] .= newModel(inputs)
    end
    outputs = softmax(outputs')'
    vmax = maximum(outputs, dims=2)
    outputs = (outputs .== vmax)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets)
    acc = accuracy(vec(outputs), vec(targets))
    errorRate = 1 - acc
    recall = mean(outputs[targets])
    specificity = mean(.!outputs[.!targets])
    precision = mean(targets[outputs])
    NPV = mean(.!targets[.!outputs])
    if isnan(recall) && isnan(precision)
        recall = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(NPV)
        specificity = 1.0
        NPV = 1.0
    end
    recall = isnan(recall) ? 0.0 : recall
    specificity = isnan(specificity) ? 0.0 : specificity
    precision = isnan(precision) ? 0.0 : precision
    NPV = isnan(NPV) ? 0.0 : NPV
    F1 = (recall == precision == 0.0) ? 0.0 : 2 * (recall * precision) / (recall + precision)
    confMatrix = Array{Int64}(undef, 2, 2)
    confMatrix[1, 1] = sum(.!targets .& .!outputs) # TN
    confMatrix[1, 2] = sum(targets .& outputs) # FP
    confMatrix[2, 1] = sum(targets .& .!outputs) # FN
    confMatrix[2, 2] = sum(targets .& outputs) # TP
    if weighted
        wTP = sum(targets .& outputs, dims=2)
        wFN = sum(targets .& .!outputs, dims=2)
        wFP = sum(.!targets .& outputs, dims=1)
        wTN = sum(.!targets .& .!outputs, dims=1)
        wAccuracy = sum(wTP) / (sum(wTP) + sum(wFN))
        wErrorRate = 1 - wAccuracy
        wPrecision = sum(wTP) / (sum(wTP) + sum(wFP))
        wRecall = sum(wTP) / (sum(wTP) + sum(wFN))
        wF1 = 2 * (wPrecision * wRecall) / (wPrecision + wRecall)
        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix, wAccuracy, wErrorRate, wPrecision, wRecall, wF1)
    end
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

# Se a añadido la linea outputs = round.(outputs)
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets)
    outputs = round.(outputs)
    acc = accuracy(vec(outputs), vec(targets))
    errorRate = 1 - acc
    recall = mean(outputs[targets])
    specificity = mean(.!outputs[.!targets])
    precision = mean(targets[outputs])
    NPV = mean(.!targets[.!outputs])
    if isnan(recall) && isnan(precision)
        recall = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(NPV)
        specificity = 1.0
        NPV = 1.0
    end
    recall = isnan(recall) ? 0.0 : recall
    specificity = isnan(specificity) ? 0.0 : specificity
    precision = isnan(precision) ? 0.0 : precision
    NPV = isnan(NPV) ? 0.0 : NPV
    F1 = (recall == precision == 0.0) ? 0.0 : 2 * (recall * precision) / (recall + precision)
    confMatrix = Array{Int64}(undef, 2, 2)
    confMatrix[1, 1] = sum(.!targets .& .!outputs) # TN
    confMatrix[1, 2] = sum(targets .& outputs) # FP
    confMatrix[2, 1] = sum(targets .& .!outputs) # FN
    confMatrix[2, 2] = sum(targets .& outputs) # TP
    if weighted
        wTP = sum(targets .& outputs, dims=2)
        wFN = sum(targets .& .!outputs, dims=2)
        wFP = sum(.!targets .& outputs, dims=1)
        wTN = sum(.!targets .& .!outputs, dims=1)
        wAccuracy = sum(wTP) / (sum(wTP) + sum(wFN))
        wErrorRate = 1 - wAccuracy
        wPrecision = sum(wTP) / (sum(wTP) + sum(wFP))
        wRecall = sum(wTP) / (sum(wTP) + sum(wFN))
        wF1 = 2 * (wPrecision * wRecall) / (wPrecision + wRecall)
        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix, wAccuracy, wErrorRate, wPrecision, wRecall, wF1)
    end
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

# Convertimos los outputs y los targets en vectores booleanos y cambiamos el calculo ponderado para que use dims=1
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert length(outputs) == length(targets)
    outputs = convert(Vector{Bool}, outputs)
    targets = convert(Vector{Bool}, targets)
    acc = accuracy(outputs, targets)
    errorRate = 1 - acc
    recall = mean(outputs[targets])
    specificity = mean(.!outputs[.!targets])
    precision = mean(targets[outputs])
    NPV = mean(.!targets[.!outputs])
    if isnan(recall) && isnan(precision)
        recall = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(NPV)
        specificity = 1.0
        NPV = 1.0
    end
    recall = isnan(recall) ? 0.0 : recall
    specificity = isnan(specificity) ? 0.0 : specificity
    precision = isnan(precision) ? 0.0 : precision
    NPV = isnan(NPV) ? 0.0 : NPV
    F1 = (recall == precision == 0.0) ? 0.0 : 2 * (recall * precision) / (recall + precision)
    confMatrix = Array{Int64}(undef, 2, 2)
    confMatrix[1, 1] = sum(.!targets .& .!outputs) # TN
    confMatrix[1, 2] = sum(targets .& outputs) # FP
    confMatrix[2, 1] = sum(targets .& .!outputs) # FN
    confMatrix[2, 2] = sum(targets .& outputs) # TP
    if weighted
        wTP = sum(targets .& outputs, dims=1)
        wFN = sum(targets .& .!outputs, dims=1)
        wFP = sum(.!targets .& outputs, dims=1)
        wTN = sum(.!targets .& .!outputs, dims=1)
        wAccuracy = sum(wTP) / (sum(wTP) + sum(wFN))
        wErrorRate = 1 - wAccuracy
        wPrecision = sum(wTP) / (sum(wTP) + sum(wFP))
        wRecall = sum(wTP) / (sum(wTP) + sum(wFN))
        wF1 = 2 * (wPrecision * wRecall) / (wPrecision + wRecall)
        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix, wAccuracy, wErrorRate, wPrecision, wRecall, wF1)
    end
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, sensibilidad, especifidad, precision, NPV, F1, confMatrix, wAccuracy, wErrorRate, wPrecision, wRecall, wF1) = confusionMatrix(outputs, targets)
    println("La precisión es:", acc)
    println("La tasa de error es:", errorRate)
    println("La sensibilidad es:", sensibilidad)
    println("La especificidad es:", especifidad)
    println("La VPP es:", precision)
    println("La VPV es:", NPV)
    println("El f1-score es:", F1)
    for i in 1:size(confMatrix, 1)[1]
        print(confMatrix[i, :])
        print("\n")
    end
    return (acc, errorRate, recall, specificity, precision, NPV, F1,
        confMatrix)
end

printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    weighted::Bool=true) = printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)

# ----------------------------------------P5-----------------------------------------

using Random
using Random: seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N / k)))
    indices = indices[1:N]
    shuffle!(indices)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)
    indices = repeat(1:k, Int64(ceil(N / k)))
    indices = indices[1:N]
    shuffle!(indices)
    subIndices = fill(0, N)
    for i in 1:k
        classIndices = findall(targets[:, i])
        subSize = Int64(ceil(length(classIndices) / k))
        subIndices[classIndices] = indices[(i-1)*subSize+1:min(i * subSize, length(classIndices))]
    end
    return subIndices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)
    indices = fill(0, N)
    for i in 1:size(targets, 2)
        classIndices = findall(targets[:, i])
        subSize = Int64(ceil(length(classIndices) / k))
        indices[classIndices] .= crossvalidation(ones(length(classIndices)), k)[1:length(classIndices)]
    end
    return indices
end

# La función realiza la codificación one-hot de un vector de etiquetas, 
# creando una matriz booleana donde cada columna representa una etiqueta y las filas son los patrones, 
# y luego aplica la validación cruzada estratificada.

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    N = length(targets)
    classDict = Dict(unique(targets) .=> 1:length(unique(targets)))
    targetsNumeric = [classDict[t] for t in targets]
    targetsBool = falses(N, length(classDict))
    for i in 1:length(classDict)
        targetsBool[:, i] = targetsNumeric .== i
    end
    indices = crossvalidation(targetsBool, k)
    return indices
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)




end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

end
# ----------------------------------------P6-----------------------------------------

# Pkg.add("ScikitLearn")) y los modelos (svm, tree, neighbors)

using ScikitLearn
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier

#model = SVC(kernel="rbf", degree=3, gamma=2, C=1);
#model = DecisionTreeClassifier(max_depth=4, random_state=1)
#model = KNeighborsClassifier(3);

#Ejemplo de uso (las salidas son un vector)
#fit!(model, trainingInputs, trainingTargets); 

#Una vez entrenado se pueden predecir las soluciones usando predict
#testOutputs = predict(model, testInputs);

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

    # Verificar que la cantidad de patrones de entrada coincide con la cantidad de objetivos.
    @assert(size(inputs, 1) == length(targets))

    # Determinar las clases únicas de los objetivos.
    classes = unique(targets)

    # Codificar los objetivos de salida deseados si el modelo es una red neuronal artificial (ANN).
    if modelType == :ANN
        targets = oneHotEncoding(targets, classes)
    end

    # Obtener el número de pliegues.
    numFolds = maximum(crossValidationIndices)

    # Inicializar vectores para almacenar las métricas de evaluación del modelo.
    testAccuracies = Array{Float64,1}(undef, numFolds)
    testF1 = Array{Float64,1}(undef, numFolds)

    # Para cada pliegue, entrenar y evaluar el modelo.
    for numFold in 1:numFolds
        # Dividir los datos en conjuntos de entrenamiento y prueba.
        trainingInputs = inputs[crossValidationIndices.!=numFold, :]
        testInputs = inputs[crossValidationIndices.==numFold, :]
        trainingTargets = targets[crossValidationIndices.!=numFold]
        testTargets = targets[crossValidationIndices.==numFold]

        # Entrenar y evaluar el modelo en función de su tipo.
        if (modelType == :SVM) || (modelType == :DecisionTree) || (modelType == :kNN)
            model = train_and_evaluate_non_ann_model(modelType, modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
        else
            @assert(modelType == :ANN)
            model = train_and_evaluate_ann_model(modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
        end

        # Calcular y almacenar las métricas para el pliegue actual.
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(model[:testOutputs], testTargets)
        testAccuracies[numFold] = acc
        testF1[numFold] = F1
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100 * testAccuracies[numFold], " %, F1: ", 100 * testF1[numFold], " %")
    end

    # Imprimir las métricas promedio para todos los pliegues.
    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100 * mean(testAccuracies), ", with a standard desviation of ", 100 * std(testAccuracies))
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100 * mean(testF1), ", with a standard desviation of ", 100 * std(testF1))

    # Devolver una tupla que contiene las métricas promedio y la desviación estándar.
    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1))
end
# Función para entrenar y evaluar modelos de aprendizaje automático no basados en redes neuronales artificiales.
function train_and_evaluate_non_ann_model(modelType, modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
    if modelType == :SVM
        # Crear un modelo de SVM con los hiperparámetros especificados.
        model = SVC(
            kernel=modelHyperparameters["kernel"],
            degree=modelHyperparameters["kernelDegree"],
            gamma=modelHyperparameters["kernelGamma"],
            C=modelHyperparameters["C"]
        )
    elseif modelType == :DecisionTree
        # Crear un modelo de árbol de decisión con el parámetro de profundidad máxima especificado.
        model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1)
    elseif modelType == :kNN
        # Crear un modelo kNN con el número de vecinos especificado.
        model = KNeighborsClassifier(modelHyperparameters["numNeighbors"])
    end

    # Ajustar el modelo a los datos de entrenamiento.
    model = fit!(model, trainingInputs, trainingTargets)

    # Hacer predicciones con el modelo sobre los datos de prueba.
    testOutputs = predict(model, testInputs)

    # Devolver un diccionario que contiene las predicciones del modelo sobre los datos de prueba.
    return Dict([(:testOutputs, testOutputs)])
end

# Función para entrenar y evaluar modelos de redes neuronales artificiales (ANN).
function train_and_evaluate_ann_model(modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
    # Obtener el número de ejecuciones a realizar.
    numExecutions = modelHyperparameters["numExecutions"]

    # Inicializar un vector para almacenar las exactitudes de las diferentes ejecuciones.
    testAccuraciesEachRepetition = Array{Float64,1}(undef, numExecutions)
    for numTraining in 1:numExecutions
        # Realizar la validación cruzada si el parámetro "validationRatio" es mayor que cero.
        if modelHyperparameters["validationRatio"] > 0
            # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación.
            (trainingIndices, validationIndices) = holdOut(size(trainingInputs, 1),
                modelHyperparameters["validationRatio"] * size(trainingInputs, 1) / size(inputs, 1))

            # Entrenar una red neuronal artificial con los datos de entrenamiento y validación, y evaluarla en los datos de prueba.
            ann, = trainClassANN(modelHyperparameters["topology"],
                trainingInputs[trainingIndices, :],
                trainingTargets[trainingIndices, :],
                trainingInputs[validationIndices, :],
                trainingTargets[validationIndices, :],
                testInputs, testTargets;
                maxEpochs=modelHyperparameters["maxEpochs"],
                learningRate=modelHyperparameters["learningRate"],
                maxEpochsVal=modelHyperparameters["maxEpochsVal"])
        else
            # Entrenar una red neuronal artificial con todos los datos de entrenamiento, y evaluarla en los datos de prueba.
            ann, = trainClassANN(modelHyperparameters["topology"],
                trainingInputs, trainingTargets,
                testInputs, testTargets;
                maxEpochs=modelHyperparameters["maxEpochs"],
                learningRate=modelHyperparameters["learningRate"])
        end
        # Calcular la exactitud y la F1 para cada ejecución.
        (testAccuraciesEachRepetition[numTraining], _, _, _, _, _,
            testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets)
    end

    # Calcular la exactitud y la F1 promedio para todas las ejecuciones.
    acc = mean(testAccuraciesEachRepetition)
    F1 = mean(testF1EachRepetition)

    # Devolver un diccionario que contiene las predicciones del modelo sobre los datos de prueba, así como la exactitud y la F1 promedio sobre todas las ejecuciones.
    return Dict([(:testOutputs, collect(ann(testInputs')')), (:acc, acc), (:F1, F1)])
end



# ---------------------------------------------------Probar P6-------------------------------------------------
# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);
numFolds = 10;
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C = 1;
# Parametros del arbol de decision
maxDepth = 4;
# Parapetros de kNN
numNeighbors = 3;
# Cargamos el dataset
dataset = readdlm("iris.data", ',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:, 1:4]);
targets = dataset[:, 5];
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);
# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, numFolds);
# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);