using System;

namespace NeuralNet {
	public sealed class EliminationRegularization : Regularization {
		private readonly float _sqrAlpha;

		public EliminationRegularization(float regularizationFactor, float alpha) : base(regularizationFactor) {
			_sqrAlpha = alpha*alpha;
		}

		public override float GetDerivative(float value) {
			var tmp = _sqrAlpha + value*value;
			return 2.0f*RegularizationFactor*_sqrAlpha*value/(tmp*tmp);
		}

		public float Alpha {
			get { return (float) Math.Sqrt(_sqrAlpha); }
		}
	}
}