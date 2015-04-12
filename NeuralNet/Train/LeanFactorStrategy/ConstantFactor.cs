namespace NeuralNet.LeanFactorStrategy {
	public sealed class ConstantFactor : ILearnFactorStrategy {
		private readonly float _constValue;
		
		public ConstantFactor(float constValue = 1f) {
			_constValue = constValue;
		}

		public float ConstantValue {
			get { return _constValue; }
		}

		public float GetFactor(int iterNumber) {
			return _constValue;
		}
	}
}
