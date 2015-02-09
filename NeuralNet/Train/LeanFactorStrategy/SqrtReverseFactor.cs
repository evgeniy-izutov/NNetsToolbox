using System;

namespace NeuralNet {
	public sealed class SqrtReverseFactor : ILearnFactorStrategy {
		private const float Epsilon = 0.000001f;
		
		public float GetFactor(int iterNumber) {
			return (float) Math.Sqrt(1.0/(iterNumber + Epsilon));
		}
	}
}