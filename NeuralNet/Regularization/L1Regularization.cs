using System;

namespace NeuralNet {
	public sealed class L1Regularization : Regularization {
		public L1Regularization(float regularizationFactor) : base(regularizationFactor) {}

		public override float GetDerivative(float value) {
			return RegularizationFactor*Math.Sign(value);
		}
	}
}