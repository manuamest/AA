using FileIO;
using DelimitedFiles;
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    if (numClasses == 2)
        return reshape((feature .== classes[1]), :, 1)
    else
        encodedTargets = Array{Bool,2}(undef, size(feature, 1), numClasses)
        for i = 1:numClasses
            encodedTargets[:, i] .= (feature .== classes[i])
        end
    end
    return encodedTargets
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature););
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

using Statistics
using Flux
using Flux.Losses
#Calculate normalization parameters:
calculateMinMaxNormalizationParameters(inputs::AbstractArray{<:Real,2}) = (minimum(inputs, dims=1), maximum(inputs, dims=1));

calculateZeroMeanNormalizationParameters(inputs::AbstractArray{<:Real,2}) = (mean(inputs, dims=1), std(inputs, dims=1));

#Normalize with Min Max:
function normalizeMinMaxAux(value::Real, max::Real, min::Real)
    if (max == min)
        return 0
    else
        return (value - min) / (max - min)
    end
end;

function normalizeMinMax!(inputs::AbstractArray{<:Real,2}, minMax::NTuple{2,AbstractArray{<:Real,2}})
    minMatrix = minMax[1]
    maxMatrix = minMax[2]
    inputs = normalizeMinMaxAux.(inputs, maxMatrix, minMatrix)
end;

normalizeMinMax!(inputs::AbstractArray{<:Real,2}) = normalizeMinMax!(inputs, calculateMinMaxNormalizationParameters(inputs));

function normalizeMinMax(inputs::AbstractArray{<:Real,2}, minMax::NTuple{2,AbstractArray{<:Real,2}})
    normalizedInputs = copy(inputs)
    normalizeMinMax!(normalizedInputs, minMax)
    return normalizedInputs
end;

function normalizeMinMax(inputs::AbstractArray{<:Real,2})
    normalizedInputs = copy(inputs)
    normalizeMinMax!(normalizedInputs)
    return normalizedInputs
end;

#Normalize with Zero Mean:
function normalizeZeroMeanAux(value::Real, mn::Real, stdev::Real)
    if (stdev == 0)
        return 0
    else
        return (value - mn) / stdev
    end
end;

function normalizeZeroMean!(inputs::AbstractArray{<:Real,2}, meanStd::NTuple{2,AbstractArray{<:Real,2}})
    meanMatrix = meanStd[1]
    stdMatrix = meanStd[2]
    inputs = normalizeZeroMeanAux.(inputs, meanMatrix, stdMatrix)
end;

normalizeZeroMean!(inputs::AbstractArray{<:Real,2}) = normalizeMinMax!(inputs, calculateMinMaxNormalizationParameters(inputs));

function normalizeZeroMean(inputs::AbstractArray{<:Real,2}, meanStd::NTuple{2,AbstractArray{<:Real,2}})
    normalizedInputs = copy(inputs)
    normalizeZeroMean!(normalizedInputs, meanStd)
    return normalizedInputs
end;

function normalizeZeroMean(inputs::AbstractArray{<:Real,2})
    normalizedInputs = copy(inputs)
    normalizeZeroMean!(normalizedInputs)
    return normalizedInputs
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Real=0.5)
    numColumns = size(outputs, 2)
    if numColumns == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end
end;

accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1}) = mean(outputs .== targets);


function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    numColumns1 = size(targets, 2)
    numColumns2 = size(outputs, 2)
    @assert(numColumns1 == numColumns2)
    if numColumns1 == 1
        return accuracy(targets[:], outputs[:])
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        return mean(correctClassifications)
    end
end;

accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, threshold::Real=0.5) = accuracy(targets, classifyOutputs(outputs, threshold));

accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2}, threshold::Real=0.5) = accuracy(classifyOutputs(targets, threshold), outputs);
# -------------------------------------------------------
# Funciones para crear y entrenar una RNA
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    init = 1
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[init]))
        numInputsLayer = numOutputsLayer
        init += 1
    end
    if numOutputs == 1
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
# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar
# Para calcular los parametros de normalizacion, segun la forma de normalizar 
#que se desee:

# -------------------------------------------------------------------------

using Random
#holdOut que dados dos parámetros, N (igual al número de patrones) y P (valor entre 0 y 1, que indica el porcentaje de patrones que
#se separarán para el conjunto de test)
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

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(length(outputs) == length(targets))
    # Para calcular la precision y la tasa de error, se puede llamar a las 
    #funciones definidas en la practica 2
    acc = accuracy(outputs, targets) # Precision, definida previamente 
    #en una practica anterior
    errorRate = 1.0 - acc
    recall = mean(outputs[targets]) # Sensibilidad
    specificity = mean(.!outputs[.!targets]) # Especificidad
    precision = mean(targets[outputs]) # Valor predictivo positivo
    NPV = mean(.!targets[.!outputs]) # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN
    # Para el caso de sensibilidad y especificidad, en un conjunto de 
    #entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado
    #entrenar con una unica clase
    # Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un 
    #conjunto de test, si este sólo tiene patrones de una clase
    # Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya 
    #clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que 
    #no se han podido evaluar
    # Sin embargo, como es posible que se quiera combinar estos valores al 
    #evaluar una clasificacion multiclase, es necesario asignarles un valor. El 
    #criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o
    #los VN son el 100% de los patrones

    # En este caso, el sistema ha actuado correctamente, así que controlamos 
    #primero este caso
    if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
        recall = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
        specificity = 1.0
        NPV = 1.0
    end
    # Ahora controlamos los casos en los que no se han podido evaluar las 
    #metricas excluyendo los casos anteriores
    recall = isnan(recall) ? 0.0 : recall
    specificity = isnan(specificity) ? 0.0 : specificity
    precision = isnan(precision) ? 0.0 : precision
    NPV = isnan(NPV) ? 0.0 : NPV
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero 
    #no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces 
    #VPP=0 y viceversa
    F1 = (recall == precision == 0.0) ? 0.0 :
         2 * (recall * precision) / (recall + precision)
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2)
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las 
    #columnas los clasificados (outputs)
    # Primera fila/columna: negativos
    # Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o 
    #positivos
    confMatrix[1, 1] = sum(.!targets .& .!outputs) # VN
    confMatrix[1, 2] = sum(.!targets .& outputs) # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o 
    #positivos
    confMatrix[2, 1] = sum(targets .& .!outputs) # FN
    confMatrix[2, 2] = sum(targets .& outputs) # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
    threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs .>= threshold),
    targets);

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    bool_outputs = outputs .>= threshold
    return confusionMatrix(bool_outputs, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2};
    weighted::Bool=true)
    @assert(size(outputs) == size(targets))
    numClasses = size(targets, 2)
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses != 2)
    if (numClasses == 1)
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    else
        # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
        @assert(all(sum(outputs, dims=2) .== 1))
        # Reservamos memoria para las metricas de cada clase, inicializandolas a

        #0 porque algunas posiblemente no se calculen
        recall = zeros(numClasses)
        specificity = zeros(numClasses)
        precision = zeros(numClasses)
        NPV = zeros(numClasses)
        F1 = zeros(numClasses)
        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses)
        # Calculamos el numero de patrones de cada clase
        numInstancesFromEachClass = vec(sum(targets, dims=1))
        # Calculamos las metricas para cada clase, esto se haria con un bucle 
        #similar a "for numClass in 1:numClasses" que itere por todas las clases
        # Sin embargo, solo hacemos este calculo para las clases que tengan 
        #algun patron
        # Puede ocurrir que alguna clase no tenga patrones como consecuencia de
        #haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
        # En aquellas clases en las que no haya patrones, los valores de las 
        #metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a 
        #la hora de unir estas metricas
        for numClass in findall(numInstancesFromEachClass .> 0)
            # Calculamos las metricas de cada problema binario correspondiente a
            #cada clase y las almacenamos en los vectores correspondientes
            (_, _, recall[numClass], specificity[numClass], precision[numClass],
                NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:, numClass],
                targets[:, numClass])
        end
        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses)
        # Calculamos la matriz de confusión haciendo un bucle doble que itere 
        #sobre las clases
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            # Igual que antes, ponemos en las filas los que pertenecen a cada 
            #clase (targets) y en las columnas los clasificados (outputs)
            confMatrix[numClassTarget, numClassOutput] =
                sum(targets[:, numClassTarget] .& outputs[:, numClassOutput])
        end
        # Aplicamos las forma de combinar las metricas macro o weighted
        if weighted
            # Calculamos los valores de ponderacion para hacer el promedio
            weights = numInstancesFromEachClass ./ sum(numInstancesFromEachClass)
            recall = sum(weights .* recall)
            specificity = sum(weights .* specificity)
            precision = sum(weights .* precision)
            NPV = sum(weights .* NPV)
            F1 = sum(weights .* F1)
        else
            # No realizo la media tal cual con la funcion mean, porque puede 
            #+haber clases sin instancias
            # En su lugar, realizo la media solamente de las clases que tengan 
            #instancias
            numClassesWithInstances = sum(numInstancesFromEachClass .> 0)
            recall = sum(recall) / numClassesWithInstances

            specificity = sum(specificity) / numClassesWithInstances
            precision = sum(precision) / numClassesWithInstances
            NPV = sum(NPV) / numClassesWithInstances
            F1 = sum(F1) / numClassesWithInstances
        end
        # Precision y tasa de error las calculamos con las funciones definidas 
        #previamente
        acc = accuracy(outputs, targets)
        errorRate = 1 - acc
        return (acc, errorRate, recall, specificity, precision, NPV, F1,
            confMatrix)
    end
end;

confusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets;
    weighted=weighted);

function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de 
    #las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]))
    classes = unique(targets)
    # Es importante calcular el vector de clases primero y pasarlo como 
    #argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea 
    #el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes),
        oneHotEncoding(targets, classes); weighted=weighted)
end;

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las
#metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2};
    weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) =
        confusionMatrix(outputs, targets; weighted=weighted)
    numClasses = size(confMatrix, 1)
    writeHorizontalLine() = (for i in 1:numClasses+1
        print("--------")
    end;
    println(""))
    writeHorizontalLine()
    print("\t| ")

    if (numClasses == 2)
        println(" - \t + \t|")
    else
        print.("Cl. ", 1:numClasses, "\t| ")
    end
    println("")
    writeHorizontalLine()
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses == 2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ")
        else
            print("Cl. ", numClassTarget, "\t| ")
        end
        print.(confMatrix[numClassTarget, :], "\t| ")
        println("")
        writeHorizontalLine()
    end
    println("Accuracy: ", acc)
    println("Error rate: ", errorRate)
    println("Recall: ", recall)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("Negative predictive value: ", NPV)
    println("F1-score: ", F1)
    return (acc, errorRate, recall, specificity, precision, NPV, F1,
        confMatrix)
end;

printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =
    printConfusionMatrix(classifyOutputs(outputs), targets;
        weighted=weighted)

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println(confMatrix)
    println("Accuracy: $acc")
    println("Error Rate: $errorRate")
    println("Recall (Sensitivity): $recall")
    println("Specificity: $specificity")
    println("Precision (Positive Predictive Value): $precision")
    println("Negative Predictive Value: $NPV")
    println("F1 Score: $F1")
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    bool_outputs = outputs .>= threshold
    printConfusionMatrix(bool_outputs, targets)
end


function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
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
#esta es la función a arreglar, la de arriba tomadla como ejemplo, pero leed el puto pdf inútiles
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

    numFolds = length(unique(kFoldIndices))
    testAccuracies = Array{Float64,1}(undef, numFolds)
    testF1 = Array{Float64,1}(undef, numFolds)
    for numFold in 1:numFolds
        local trainingInputs, testInputs, trainingTargets, testTargets
        trainingInputs = inputs[crossValidationIndices.!=numFold, :]
        testInputs = inputs[crossValidationIndices.==numFold, :]
        trainingTargets = targets[crossValidationIndices.!=numFold, :]
        testTargets = targets[crossValidationIndices.==numFold, :]

        #Vectores adicionales para almacenar las métricas de cada entrenamiento
        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining)
        testF1EachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining)
        for numTraining in 1:numRepetitionsAANTraining
            if validationRatio > 0

                local trainingIndices, validationIndices
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs, 1),
                    validationRatio * size(trainingInputs, 1) / size(inputs, 1))
                local ann
                ann, = trainClassANN(topology,
                    trainingInputs[trainingIndices, :],
                    trainingTargets[trainingIndices, :],
                    trainingInputs[validationIndices, :],
                    trainingTargets[validationIndices, :],
                    testInputs, testTargets;
                    maxEpochs=numMaxEpochs, learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal)

            else
                local ann
                ann, = trainClassANN(topology,
                    trainingInputs, trainingTargets,
                    testInputs, testTargets;
                    maxEpochs=numMaxEpochs, learningRate=learningRate)
            end
            (acc, _, _, _, _, _, F1, _) =
                confusionMatrix(collect(ann(testInputs')'), testTargets)
            testAccuraciesEachRepetition[numTraining] = acc
            testF1EachRepetition[numTraining] = F1
        end
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition)
        testF1[numFold] = mean(testF1EachRepetition)
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ",
            100 * testAccuracies[numFold], " %, F1: ", 100 * testF1[numFold], " %")

    end
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ",
        100 * mean(testAccuracies), ", with a standard deviation of ",
        100 * std(testAccuracies))
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ",
        100 * mean(testF1), ", with a standard deviation of ", 100 * std(testF1))

end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    trainClassANN(topology, (trainingDataset[1], reshape(trainingDataset[2], size(trainingDataset[2], 1), 1)),
        transferFunctions, maxEpochs, minLoss, learningRate, numRepetitionsAANTraining, validationRatio, maxEpochsVal)

end



function trainClassANN(topology::Array{Int64,1},
    trainingInputs::Array{Float64,2}, trainingTargets::Array{Bool,2},
    validationInputs::Array{Float64,2}, validationTargets::Array{Bool,2},
    testInputs::Array{Float64,2}, testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1,
    maxEpochsVal::Int64=6, showText::Bool=false)
    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validation como test
    @assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    @assert(size(validationInputs, 1) == size(validationTargets, 1))
    @assert(size(testInputs, 1) == size(testTargets, 1))
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento, validacion y test

    @assert(size(trainingInputs, 2) == size(validationInputs, 2) == size(testInputs, 2))

    @assert(size(trainingTargets, 2) == size(validationTargets, 2) == size(testTargets, 2))

    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs, 2), topology,
        size(trainingTargets, 2))
    # Definimos la funcion de loss
    loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) :
                 Losses.crossentropy(ann(x), y)
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[]
    trainingAccuracies = Float64[]
    validationLosses = Float64[]
    validationAccuracies = Float64[]
    testLosses = Float64[]
    testAccuracies = Float64[]
    # Empezamos en el ciclo 0
    numEpoch = 0
    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets')
        validationLoss = loss(validationInputs', validationTargets')
        testLoss = loss(testInputs', testTargets')
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs')
        validationOutputs = ann(validationInputs')
        testOutputs = ann(testInputs')
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        # Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc = accuracy(trainingOutputs,
            Array{Bool,2}(trainingTargets'))
        validationAcc = accuracy(validationOutputs,
            Array{Bool,2}(validationTargets'))
        testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets'))
        # Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'),
            trainingTargets)
        validationAcc = accuracy(Array{Float64,2}(validationOutputs'),
            validationTargets)
        testAcc = accuracy(Array{Float64,2}(testOutputs'),
            testTargets)
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ",
           accuracy: ", 100 * trainingAcc, " % - Validation loss: ", validationLoss, ",
             accuracy: ", 100 * validationAcc, " % - Test loss: ", testLoss, ", accuracy: ",
                100 * testAcc, " %")
        end
        return (trainingLoss, trainingAcc, validationLoss, validationAcc,
            testLoss, testAcc)
    end
    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics()
    # y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss)
    push!(trainingAccuracies, trainingAccuracy)
    push!(validationLosses, validationLoss)
    push!(validationAccuracies, validationAccuracy)
    push!(testLosses, testLoss)
    push!(testAccuracies, testAccuracy)
    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0
    bestValidationLoss = validationLoss
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann)
    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas(cada  patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')],
            ADAM(learningRate))
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy,
            testLoss, testAccuracy) = calculateMetrics()
        # y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)
        push!(validationLosses, validationLoss)
        push!(validationAccuracies, validationAccuracy)
        push!(testLosses, testLoss)
        push!(testAccuracies, testAccuracy)
        # Aplicamos la parada temprana
        if (validationLoss < bestValidationLoss)
            bestValidationLoss = validationLoss
            numEpochsValidation = 0
            bestANN = deepcopy(ann)
        else
            numEpochsValidation += 1
        end
    end
    return (bestANN, trainingLosses, validationLosses, testLosses,
        trainingAccuracies, validationAccuracies, testAccuracies)
end;

function trainClassANN(topology::Array{Int64,1},
    trainingInputs::Array{Float64,2}, trainingTargets::Array{Bool,2},
    testInputs::Array{Float64,2}, testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1,
    showText::Bool=false)
    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en 
    #entrenamiento como en test
    @assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    @assert(size(testInputs, 1) == size(testTargets, 1))
    # Comprobamos que el numero de columnas coincide en los grupos de 
    #entrenamiento y test
    @assert(size(trainingInputs, 2) == size(testInputs, 2))
    @assert(size(trainingTargets, 2) == size(testTargets, 2))
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs, 2), topology,
        size(trainingTargets, 2))
    # Definimos la funcion de loss
    loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) :
                 Losses.crossentropy(ann(x), y)
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[]
    trainingAccuracies = Float64[]
    testLosses = Float64[]
    testAccuracies = Float64[]
    # Empezamos en el ciclo 0
    numEpoch = 0
    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar 
        #las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets')
        testLoss = loss(testInputs', testTargets')

        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay 
        #que pasar la matriz de entradas traspuesta (cada patron en una columna). La 
        #matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs')
        testOutputs = ann(testInputs')

        # Convertimos las matrices de salida de Float64 a Bool
        trainingOutputsBool = convert(Matrix{Bool}, trainingOutputs .> 0.5)
        testOutputsBool = convert(Matrix{Bool}, testOutputs .> 0.5)

        # Ahora llamamos a la función accuracy con las matrices convertidas a Bool
        trainingAcc = accuracy(trainingOutputsBool, Array{Bool,2}(trainingTargets'))
        testAcc = accuracy(testOutputsBool, Array{Bool,2}(testTargets'))

        # Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        # Convertimos las matrices de salida de Float64 a Bool
        trainingOutputsBool = convert(Matrix{Bool}, trainingOutputs' .> 0.5)
        testOutputsBool = convert(Matrix{Bool}, testOutputs' .> 0.5)

        # Ahora llamamos a la función accuracy con las matrices convertidas a Bool
        trainingAcc = accuracy(trainingOutputsBool, trainingTargets)
        testAcc = accuracy(testOutputsBool, testTargets)
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", 
           accuracy: ", 100 * trainingAcc, " % - Test loss: ", testLoss, ", accuracy: ",
                100 * testAcc, " %")
        end
        return (trainingLoss, trainingAcc, testLoss, testAcc)
    end
    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, testLoss, testAccuracy) =
        calculateMetrics()
    # y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss)
    push!(testLosses, testLoss)
    push!(trainingAccuracies, trainingAccuracy)
    push!(testAccuracies, testAccuracy)
    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')],
            ADAM(learningRate))
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) =
            calculateMetrics()
        # y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)
        push!(testLosses, testLoss)
        push!(testAccuracies, testAccuracy)
    end
    return (ann, trainingLosses, testLosses, trainingAccuracies, testAccuracies)
end;







using ScikitLearn
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})
    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs, 1) == length(targets))
    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion 
    #one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo 
    #siempre
    classes = unique(targets)
    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType == :ANN
        targets = oneHotEncoding(targets, classes)
    end
    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser
    #distintas
    testAccuracies = Array{Float64,1}(undef, numFolds)
    testF1 = Array{Float64,1}(undef, numFolds)
    # Para cada fold, entrenamos
    for numFold in 1:numFolds
        # Si vamos a usar unos de estos 3 modelos
        ann = nothing
        if (modelType == :SVM) || (modelType == :DecisionTree) || (modelType == :kNN)
            # Dividimos los datos en entrenamiento y test
            trainingInputs = inputs[crossValidationIndices.!=numFold, :]
            testInputs = inputs[crossValidationIndices.==numFold, :]
            trainingTargets = targets[crossValidationIndices.!=numFold]
            testTargets = targets[crossValidationIndices.==numFold]
            if modelType == :SVM
                model = SVC(kernel=modelHyperparameters["kernel"],
                    degree=modelHyperparameters["kernelDegree"],
                    gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"])

            elseif modelType == :DecisionTree
                model =
                    DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"],
                        random_state=1)
            elseif modelType == :kNN
                model =
                    KNeighborsClassifier(modelHyperparameters["numNeighbors"])
            end
            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets)
            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs)
            trainingOutputs = predict(model, trainingInputs)
            # Calculamos las metricas correspondientes con la funcion 
            #desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs,
                testTargets)
            println("Results in the training set:")
            printConfusionMatrix(oneHotEncoding(trainingOutputs), oneHotEncoding(trainingTargets))
            println("Results in the test set:")
            printConfusionMatrix(oneHotEncoding(testOutputs), oneHotEncoding(testTargets))
        else
            # Vamos a usar RR.NN.AA.
            @assert(modelType == :ANN)
            # Dividimos los datos en entrenamiento y test
            trainingInputs = inputs[crossValidationIndices.!=numFold, :]
            testInputs = inputs[crossValidationIndices.==numFold, :]
            trainingTargets = targets[crossValidationIndices.!=numFold, :]
            testTargets = targets[crossValidationIndices.==numFold, :]
            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que 
            #entrenar varias veces, y
            # se crean vectores adicionales para almacenar las metricas para 
            #cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef,
                modelHyperparameters["numExecutions"])
            testF1EachRepetition = Array{Float64,1}(undef,
                modelHyperparameters["numExecutions"])
            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"] > 0
                    # Para el caso de entrenar una RNA con conjunto de 
                    #validacion, hacemos una división adicional:
                    # dividimos el conjunto de entrenamiento en 
                    #ntrenamiento+validacion
                    # Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) =
                        holdOut(size(trainingInputs, 1),
                            modelHyperparameters["validationRatio"] * size(trainingInputs, 1) / size(inputs, 1))
                    # Con estos indices, se pueden crear los vectores finales 

                    #que vamos a usar para entrenar una RNA
                    # Entrenamos la RNA, teniendo cuidado de codificar las 
                    #salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        (trainingInputs[trainingIndices, :],
                            trainingTargets[trainingIndices, :]),
                        (trainingInputs[validationIndices, :],
                            trainingTargets[validationIndices, :]),
                        (testInputs, testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"],
                        learningRate=modelHyperparameters["learningRate"],
                        maxEpochsVal=modelHyperparameters["maxEpochsVal"])
                else
                    # Si no se desea usar conjunto de validacion, se entrena 
                    #unicamente con conjuntos de entrenamiento y test,
                    # teniendo cuidado de codificar las salidas deseadas 
                    #correctamente
                    trainingTargetsBool = Matrix{Bool}(trainingTargets)
                    testTargetsBool = Matrix{Bool}(testTargets)

                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs, trainingTargetsBool,
                        testInputs, testTargetsBool,
                        maxEpochs=modelHyperparameters["maxEpochs"],
                        learningRate=modelHyperparameters["learningRate"])


                end
                # Calculamos las metricas correspondientes con la funcion 
                #desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _,
                    testF1EachRepetition[numTraining], _) =
                    confusionMatrix(collect(ann(testInputs')'), testTargets)
            end
            # Calculamos el valor promedio de todos los entrenamientos de este 
            #fold
            println("Results in the training set:")
            printConfusionMatrix(collect(ann(trainingInputs')'), trainingTargets)
            println("Results in the test set:")
            printConfusionMatrix(collect(ann(testInputs')'), testTargets)
            acc = mean(testAccuraciesEachRepetition)
            F1 = mean(testF1EachRepetition)
        end

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc
        testF1[numFold] = F1
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:
       ", 100 * testAccuracies[numFold], " %, F1: ", 100 * testF1[numFold], " %")
    end # for numFold in 1:numFolds
    println(modelType, ": Average test accuracy on a ", numFolds, "-fold 
   crossvalidation: ", 100 * mean(testAccuracies), ", with a standard desviation of ", 100 * std(testAccuracies))
    println(modelType, ": Average test F1 on a ", numFolds, "-fold 
   crossvalidation: ", 100 * mean(testF1), ", with a standard desviation of ",
        100 * std(testF1))
    return (mean(testAccuracies), std(testAccuracies), mean(testF1),
        std(testF1))
end;



# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);
numFolds = 10;
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [15, 20]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda 
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. 
#Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el 
#conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA 
#para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.0001;
C = 100000
# Parametros del arbol de decision
maxDepth = 6
# Parapetros de kNN
numNeighbors = 6;
# Cargamos el dataset
dataset = readdlm("zombie-nozombie.data", ',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:, 1:2]);
targets = dataset[:, 3];
crossValidationIndices = crossvalidation(size(inputs, 1), numFolds);
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para 
#test
normalizeZeroMean!(inputs);
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);
# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
#println("C: ", C)
#println(", kernel: ", kernel)
#println(", kernelDegree: ", kernelDegree)
#println(", kernelGamma: ", kernelGamma)
#modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);
# Entrenamos los arboles de decision
#println("maxDepth: ", maxDepth)
#modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);
# Entrenamos los kNN
#println("numNeighbors: ", numNeighbors)
#modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);