using System;
using System.Threading.Tasks;

namespace NeuralNet {
	[Serializable]
	public sealed class SimpleNeuronBlock : BaseNeuralBlock {
		public SimpleNeuronBlock(int size, BaseNeuralBlock parent, IActivationFunction activationFunction)
			: base(size, new[] {parent}, activationFunction) {}

		public SimpleNeuronBlock(int size, int parentSize, IActivationFunction activationFunction)
			: base(size, parentSize, activationFunction) {}

		public override void Calculate() {
			var parent = Parents[0];
			var parentState = parent.GetState();
			var parentSize = parent.Size;
			var weightsForParent = Weights[0];
			var neuronsCount = State.Length;

			Parallel.For(0, neuronsCount, neuronNum => {
				var sum = 0.0f;
				for (var i = 0; i < parentState.Length; i++) {
					sum += parentState[i]
					       *weightsForParent[neuronNum*parentSize + i];
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = ActivationFunction.Calculate(sum);
			});
		}

		public override void Calculate(float[] input) {
			var firstParentBlockWeights = Weights[0];
			var inputSize = input.Length;
			var neuronsCount = State.Length;

			Parallel.For(0, neuronsCount, neuronNum => {
				var sum = 0.0f;
				for (var i = 0; i < inputSize; i++) {
					sum += input[i]*firstParentBlockWeights[neuronNum*inputSize + i];
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = ActivationFunction.Calculate(sum);
			});
		}
	}
}