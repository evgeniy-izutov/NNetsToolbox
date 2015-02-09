using System.Collections.Generic;
using GeneticAlgorithm;
using StandardTypes;

namespace NeuralNet.MultyLayerPerceptron {
	class FitnessFunctionOnDistribution : IFitnessFunction {
		private readonly MultyLayerPerceptron _neuralNet;
        private readonly IList<TrainPair> _trainingData;
		private readonly INormalizeMethod _normilazeMethod;
		private readonly float[] _distribution;
		private readonly float[] _neuronnetOutput;

		public FitnessFunctionOnDistribution(MultyLayerPerceptron neuralNet, IList<TrainPair> trainingData, INormalizeMethod normilazeMethod, float[] distribution) {
			_neuralNet = neuralNet;
			_distribution = distribution;
			_normilazeMethod = normilazeMethod;
			_trainingData = trainingData;
			_neuronnetOutput = new float[trainingData[0].Output.Length];
		}

		public void Fitness(IIndividual individual) {
			ApplyWeights(individual);
            individual.Fitness = CalcErrorOnDistribution();
            individual.IsFitnessAvailable = true;
		}

		private void ApplyWeights(IIndividual individual) {
			var chromosomes = individual.Chromosomes;
			var chromosomeIndex = 0;
			var layers = _neuralNet.Layers;
			foreach (var neuronBlock in layers) {
				neuronBlock.SetWeightsFor(0, chromosomes[chromosomeIndex++]);
				neuronBlock.SetBias(chromosomes[chromosomeIndex++]);
			}
		}

		private float CalcErrorOnDistribution() {
			var outputError = 0.0f;
            for (var i = 0; i < _trainingData.Count; i++) {
                outputError += IsErrorOnTrainingPair(_trainingData[i])*_distribution[i];
            }
            return outputError;
		}

		private float IsErrorOnTrainingPair(TrainPair trainingPair) {
			var isError = 0.0f;
			_neuralNet.Predict(trainingPair.Input, _neuronnetOutput);
			_normilazeMethod.DenormalizeOutputVector(_neuronnetOutput);

			var neuronnetAnswer = FindMaxValuePosition(_neuronnetOutput);
			var realAnswer = FindMaxValuePosition(trainingPair.Output);
			if (neuronnetAnswer != realAnswer) {
				isError = 1.0f;
			}

			return isError;
		}

		private static int FindMaxValuePosition(float[] vector) {
			var maxValuePos = 0;
			var maxValue = vector[0];
			for (var i = 1; i < vector.Length; i++) {
				var curValue = vector[i];
				if (curValue > maxValue) {
					maxValue = curValue;
					maxValuePos = i;
				}
			}
			return maxValuePos;
		}
	}
}