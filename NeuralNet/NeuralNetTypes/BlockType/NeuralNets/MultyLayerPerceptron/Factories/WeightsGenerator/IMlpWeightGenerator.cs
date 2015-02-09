namespace NeuralNet.MultyLayerPerceptron {
	public interface IMlpWeightGenerator {
		void GenerateNewWeights(MultyLayerPerceptron perceptron);
	}

	public enum Distribution {
		Uniform,
		Normal,
    }
}