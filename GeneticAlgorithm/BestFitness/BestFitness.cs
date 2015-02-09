namespace GeneticAlgorithm {
	internal sealed class BestFitness : IBestFitness {
		private float _value;
		private int _position;
		private float _totalSum;

		public BestFitness (float value, int position, float totalSum) {
			_value = value;
			_position = position;
			_totalSum = totalSum;
		}

		public float Value {
			get { return _value; }
			set { _value = value; }
		}

		public int Position {
			get { return _position; }
			set { _position = value; }
		}

		public float TotalSum {
			get { return _totalSum; }
			set { _totalSum = value; }
		}
	}
}