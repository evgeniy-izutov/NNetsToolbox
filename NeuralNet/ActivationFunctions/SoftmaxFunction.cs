using System;

namespace NeuralNet {
	[Serializable]
	public sealed class SoftmaxFunction : IActivationFunction {
		public float Calculate(float x) {
			throw new NotImplementedException();
		}

		public float CalculateFirstDerivative(float x) {
			throw new NotImplementedException();
		}

		public float CalculateFirstDerivative(float[] state, int index) {
			throw new NotImplementedException();
		}

		public void CalculateFirstDerivative(float[] target, float[] factors, float[] state) {
			for (var i = 0; i < state.Length; i++) {
				target[i] = factors[i];
			}
		}

		public void CalculateFirstDerivative(float[] target, float[] state) {
			throw new NotImplementedException();
		}

		public float GetMaxDerivativeZone(float maxValuePercent) {
			return 1f;
		}

		public float CalculateInvers(float y) {
			throw new NotImplementedException();
		}
	}
}