
using Statistics
using DelimitedFiles
dataset = readdlm("/Users/fiopans1/git/class-repositories/pracicas_AA/zombie-nozombie.data", ',');
inputs = dataset[:, 1:2];
targets = dataset[:, 3];
println(size(inputs));
println(size(targets));

