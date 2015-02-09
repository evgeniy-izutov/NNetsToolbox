using System;
using System.Collections.Generic;

namespace StandardTypes {
	[Serializable]
	public class TrainSingle : TrainData, ICopyType<TrainSingle> {
		private float[] _inputData;
		private HashSet<int> _missedInputIndexes;

		public TrainSingle() {
		}

		public TrainSingle(float[] inputData, HashSet<int> missedInputIndexes = null) {
			_inputData = inputData;
			_missedInputIndexes = missedInputIndexes;
		}

		public TrainSingle(TrainSingle source) {
			var sourceInputData = source._inputData;
			_inputData = new float[sourceInputData.Length];
			sourceInputData.CopyTo(_inputData, 0);

			_missedInputIndexes = null;
			if (source._missedInputIndexes != null) {
				_missedInputIndexes = new HashSet<int>(source._missedInputIndexes);
			}

			Id = source.Id;
			Weight = source.Weight;
		}

		public float[] Input {
			get { return _inputData; }
			set { _inputData = value; }
		}

		public int InputLength {
			get { return _inputData.Length; }
		}

		public HashSet<int> MissedInputIndexes {
			get { return _missedInputIndexes; }
			set { _missedInputIndexes = value; }
		}

		public TrainSingle Copy() {
			return new TrainSingle(this);
		}
	}
}