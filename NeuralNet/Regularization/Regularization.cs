namespace NeuralNet {
	public abstract class Regularization {
		protected float RegularizationFactor;

		protected Regularization(float regularizationFactor) {
			RegularizationFactor = regularizationFactor;
		}

		public float Factor {
			get { return RegularizationFactor; }
			set { RegularizationFactor = value; }
		}

		public abstract float GetDerivative(float value);
	}
}