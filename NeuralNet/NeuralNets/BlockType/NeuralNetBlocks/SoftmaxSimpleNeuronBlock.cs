using System;
using NeuralNet.ActivationFunctions;

namespace NeuralNet.NeuralNetBlocks {
	[Serializable]
	public sealed class SoftmaxSimpleNeuronBlock : BaseNeuralBlock {
		public SoftmaxSimpleNeuronBlock(int size, BaseNeuralBlock parent, IActivationFunction activationFunction)
			: base(size, new[] {parent}, activationFunction) {}

		public SoftmaxSimpleNeuronBlock(int size, int parentSize, IActivationFunction activationFunction)
			: base(size, parentSize, activationFunction) {}

		public override void Calculate() {
			var parent = Parents[0];
			var parentState = parent.GetState();
			var parentSize = parent.Size;
			var weightsForParent = Weights[0];
			var neuronsCount = State.Length;

			var expSum = 0.0f;
			for (var neuronNum = 0; neuronNum < neuronsCount; neuronNum++) {
				var inductionSum = 0.0f;
				for (var i = 0; i < parentState.Length; i++) {
					inductionSum += parentState[i]*weightsForParent[neuronNum*parentSize + i];
				}
				inductionSum += Bias[neuronNum];
				Net[neuronNum] = inductionSum;
				var expValue = (float) Math.Exp(inductionSum);
				expSum += expValue;
				State[neuronNum] = expValue;
			}

			for (var neuronNum = 0; neuronNum < neuronsCount; neuronNum++) {
				State[neuronNum] = State[neuronNum]/expSum;
			}
		}

		public override void Calculate(float[] input) {
			var firstParentBlockWeights = Weights[0];
			var inputSize = input.Length;
			var neuronsCount = State.Length;

			var expSum = 0.0f;
			for (var neuronNum = 0; neuronNum < neuronsCount; neuronNum++) {
				var inductionSum = 0.0f;
				for (var i = 0; i < inputSize; i++) {
					inductionSum += input[i]*firstParentBlockWeights[neuronNum*inputSize + i];
				}
				inductionSum += Bias[neuronNum];
				Net[neuronNum] = inductionSum;
				var expValue = (float) Math.Exp(inductionSum);
				expSum += expValue;
				State[neuronNum] = expValue;
			}

			for (var neuronNum = 0; neuronNum < neuronsCount; neuronNum++) {
				State[neuronNum] = State[neuronNum]/expSum;
			}
		}
	}
}
