namespace NeuralNet {
	public sealed class NoRegularization : Regularization {
		public NoRegularization() : base(0f) {}

		public override float GetDerivative(float value) {
			return 0.0f;
		}
	}
}