using System;

namespace StandardTypes.FactorStrategy {
	public sealed class SqrtReverseFactor : IFactorStrategy {
		private const float Epsilon = 0.000001f;
		private readonly float _startFactor;
		private int _localIterNumber;

		public SqrtReverseFactor(float startFactor = 1f) {
			_startFactor = startFactor;
			_localIterNumber = 1;
		}
		
		public float GetFactor(int iterNumber) {
			return (float) Math.Sqrt(1.0/(iterNumber + Epsilon))*_startFactor;
		}

		public float GetFactor() {
			var value = (float) Math.Sqrt(1.0/(_localIterNumber + Epsilon))*_startFactor;
			_localIterNumber++;
			return value;
		}
	}
}
