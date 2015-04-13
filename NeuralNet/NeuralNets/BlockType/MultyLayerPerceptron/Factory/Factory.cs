using NeuralNet.ActivationFunctions;
using NeuralNet.NeuralNetBlocks;

namespace NeuralNet.MultyLayerPerceptron {
	public sealed class Factory : INeuralNetFactory {
		private readonly int[] _layersStruct;
		private readonly IActivationFunction _hiddenLayersActivationFunction;
		private readonly IActivationFunction _outputLayerActivationFunction;
		private readonly IMlpWeightGenerator _weightGenerator;

		public Factory(int[] layersStruct, IActivationFunction hiddenActivationFunction, 
			           IActivationFunction outputActivationFunction, 
			           IMlpWeightGenerator weightGenerator) {
			_layersStruct = layersStruct;
			_hiddenLayersActivationFunction = hiddenActivationFunction;
			_outputLayerActivationFunction = outputActivationFunction;
			_weightGenerator = weightGenerator;
		}

		public INeuralNet CreateNeuralNet() {
			var layersCount = _layersStruct.Length - 1;
            var inputLayerSize = _layersStruct[0];
            var neuralNet = new MultyLayerPerceptron(layersCount);
            var lastNeuronBlock = new SimpleNeuronBlock(_layersStruct[1], inputLayerSize, _hiddenLayersActivationFunction);
            neuralNet.AddNeuralBlock(lastNeuronBlock, 0);
            for (var i = 1; i < layersCount - 1; i++) {
                lastNeuronBlock = new SimpleNeuronBlock(_layersStruct[i + 1], lastNeuronBlock, _hiddenLayersActivationFunction);
                neuralNet.AddNeuralBlock(lastNeuronBlock, i);
            }

			BaseNeuralBlock outputNeuronBlock;
			if (layersCount > 1) {
				if (_outputLayerActivationFunction is Softmax) {
					outputNeuronBlock = new SoftmaxSimpleNeuronBlock(_layersStruct[layersCount], lastNeuronBlock, new Softmax());
				}
				else {
					outputNeuronBlock = new SimpleNeuronBlock(_layersStruct[layersCount], lastNeuronBlock,
						_outputLayerActivationFunction); 
				}
			}
			else {
				if (_outputLayerActivationFunction is Softmax) {
					outputNeuronBlock = new SoftmaxSimpleNeuronBlock(_layersStruct[layersCount], inputLayerSize, new Softmax());
				}
				else {
					outputNeuronBlock = new SimpleNeuronBlock(_layersStruct[layersCount], inputLayerSize,
						_outputLayerActivationFunction); 
				}
			}
			neuralNet.AddNeuralBlock(outputNeuronBlock, layersCount - 1);

			_weightGenerator.GenerateNewWeights(neuralNet);
			
			return neuralNet;
		}
	}
}
