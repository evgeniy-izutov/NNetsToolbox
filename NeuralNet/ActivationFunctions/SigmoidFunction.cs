using System;

namespace NeuralNet {
	[Serializable]
	public sealed class SigmoidFunction : IActivationFunction {
		private readonly float _alpha;

		public SigmoidFunction(float alpha) {
			_alpha = alpha;
		}

		public float Calculate(float x) {
			return (1.0f/(1.0f + (float)Math.Exp(-_alpha*x)));
		}

		public float CalculateFirstDerivative(float x) {
			var tmp = Calculate(x);
			return _alpha*tmp*(1.0f - tmp);
		}

		public float CalculateFirstDerivative(float[] state, int index) {
			var neuronState = state[index];
			return _alpha*neuronState*(1.0f - neuronState);
		}

		public void CalculateFirstDerivative(float[] target, float[] factors, float[] state) {
			for (var i = 0; i < state.Length; i++) {
				var neuronState = state[i];
				target[i] = factors[i]*_alpha*neuronState*(1.0f - neuronState);
			}
		}

		public void CalculateFirstDerivative(float[] target, float[] state) {
			for (var i = 0; i < state.Length; i++) {
				var neuronState = state[i];
				target[i] *= _alpha*neuronState*(1.0f - neuronState);
			}
		}

		public float GetMaxDerivativeZone(float maxValuePercent) {
			var value = 1.0/Math.Sqrt(maxValuePercent);
			return (float) (2.0*Math.Log(value + Math.Sqrt(value*value - 1.0))/_alpha);
		}

		public float CalculateInvers(float y) {
			return ((float) Math.Log(y/(y - 1), Math.E))/_alpha;
		}

		public float Alpha {
			get { return _alpha; }
		}
	}
}