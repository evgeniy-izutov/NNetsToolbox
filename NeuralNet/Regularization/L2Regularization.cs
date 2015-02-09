namespace NeuralNet {
	public sealed class L2Regularization : Regularization {
		public L2Regularization(float regularizationFactor) : base(regularizationFactor) {}

		public override float GetDerivative(float value) {
			return RegularizationFactor*value;
		}
	}
}
