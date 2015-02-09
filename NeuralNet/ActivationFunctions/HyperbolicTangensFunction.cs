using System;

namespace NeuralNet {
	[Serializable]
	public sealed class HyperbolicTangensFunction : IActivationFunction {
		private readonly float _alpha;
		private readonly float _betta;
		private readonly float _derivativeFactor;

		public HyperbolicTangensFunction(float alpha, float betta) {
			_alpha = alpha;
			_betta = betta;
			_derivativeFactor = _betta/_alpha;
		}

		public float Calculate(float x) {
			return (_alpha*(float)Math.Tanh(_betta*x));
		}

		public float CalculateFirstDerivative(float x) {
			var tmp = (float)Math.Tanh(_betta*x);
			return _alpha*_betta*(1.0f - tmp*tmp);
		}

		public float CalculateFirstDerivative(float[] state, int index) {
			var neuronState = state[index];
			return _derivativeFactor*(_alpha - neuronState)*(_alpha + neuronState);
		}

		public void CalculateFirstDerivative(float[] target, float[] factors, float[] state) {
			for (var i = 0; i < state.Length; i++) {
				var neuronState = state[i];
				target[i] = factors[i]*_derivativeFactor*(_alpha - neuronState)*(_alpha + neuronState);
			}
		}

		public void CalculateFirstDerivative(float[] target, float[] state) {
			for (var i = 0; i < state.Length; i++) {
				var neuronState = state[i];
				target[i] *= _derivativeFactor*(_alpha - neuronState)*(_alpha + neuronState);
			}
		}

		public float GetMaxDerivativeZone(float maxValuePercent) {
			var value = 1.0/Math.Sqrt(maxValuePercent);
			return (float) Math.Log(value + Math.Sqrt(value*value - 1.0))/_betta;
		}

		public float CalculateInvers(float y) {
			return (float)Math.Log((_alpha + y)/(_alpha - y), (float)Math.E)/(2.0f*_betta);
		}

		public float Alpha {
			get { return _alpha; }
		}

		public float Betta {
			get { return _betta; }
		}
	}
}