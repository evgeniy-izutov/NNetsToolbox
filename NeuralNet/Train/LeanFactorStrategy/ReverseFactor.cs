namespace NeuralNet {
	public sealed class ReverseFactor : ILearnFactorStrategy {
		private const float Epsilon = 0.000001f;
		
		public float GetFactor(int iterNumber) {
			return 1f/(iterNumber + Epsilon);
		}
	}
}