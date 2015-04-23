namespace StandardTypes.FactorStrategy {
	public sealed class ReverseFactor : IFactorStrategy {
		private const float Epsilon = 0.000001f;
		private readonly float _startFactor;
		private int _localIterNumber;

		public ReverseFactor(float startFactor = 1f) {
			_startFactor = startFactor;
			_localIterNumber = 1;
		}
		
		public float GetFactor(int iterNumber) {
			return 1f/(iterNumber + Epsilon)*_startFactor;
		}

		public float GetFactor() {
			var  value = 1f/(_localIterNumber + Epsilon)*_startFactor;
			_localIterNumber++;
			return value;
		}
	}
}
