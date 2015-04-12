using System;

namespace NeuralNet.RegularizationFunctions {
	public sealed class L1 : Regularization {
		public L1(float regularizationFactor) : base(regularizationFactor) {}

		public override float GetDerivative(float value) {
			return RegularizationFactor*Math.Sign(value);
		}
	}
}
