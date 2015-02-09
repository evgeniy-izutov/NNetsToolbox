namespace NeuralNet.MultyLayerPerceptron {
	public sealed class AllZeros : IMlpWeightGenerator {
		public void GenerateNewWeights(MultyLayerPerceptron neuralNet) {
			var layers = neuralNet.Layers;
			for (var layerNum = 0; layerNum < layers.Length; layerNum++) {
				var layerSize = layers[layerNum].Size;
				var weights = layers[layerNum].GetWeights()[0];
				for (var i = 0; i < weights.Length; i++) {
					weights[i] = 0.0f;
				}
				var bias = layers[layerNum].GetBias();
				for (var i = 0; i < layerSize; i++) {
					bias[i] = 0.0f;
				}
			}
		}
	}
}