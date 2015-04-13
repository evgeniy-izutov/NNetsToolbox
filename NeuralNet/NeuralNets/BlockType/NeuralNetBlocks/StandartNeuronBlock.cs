using System;
using NeuralNet.ActivationFunctions;

namespace NeuralNet.NeuralNetBlocks {
	[Serializable]
	public sealed class StandartNeuronBlock : BaseNeuralBlock {
		public StandartNeuronBlock(int size, BaseNeuralBlock[] parents, IActivationFunction activationFunction)
			: base(size, parents, activationFunction) {}

		public StandartNeuronBlock(int size, int parentSize, IActivationFunction activationFunction)
			: base(size, parentSize, activationFunction) {}

		public override void Calculate() {
			for (var neuronNum = 0; neuronNum < State.Length; neuronNum++) {
				var sum = 0.0f;
				for (var parentNum = 0; parentNum < Parents.Length; parentNum++) {
					var parent = Parents[parentNum];
					var parentState = parent.GetState();
					var parentSize = parent.Size;
					var weightsForParent = Weights[parentNum];
					for (var i = 0; i < parentState.Length; i++) {
						sum += parentState[i]*weightsForParent[neuronNum*parentSize + i];
					}
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = ActivationFunction.Calculate(sum);
			}
		}

		public override void Calculate(float[] input) {
			var firstParentBlockWeights = Weights[0];
			var inputSize = input.Length;
			for (var neuronNum = 0; neuronNum < State.Length; neuronNum++) {
				var sum = 0.0f;
				for (var i = 0; i < inputSize; i++) {
					sum += input[i]*firstParentBlockWeights[neuronNum*inputSize + i];
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = ActivationFunction.Calculate(sum);
			}
		}
	}
}
