namespace NeuralNet.RegularizationFunctions {
	public sealed class L2 : Regularization {
		public L2(float regularizationFactor) : base(regularizationFactor) {}

		public override float GetDerivative(float value) {
			return RegularizationFactor*value;
		}
	}
}
