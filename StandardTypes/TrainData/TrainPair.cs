using System;
using System.Collections.Generic;

namespace StandardTypes {
	[Serializable]
	public sealed class TrainPair : TrainSingle, ICopyType<TrainPair> {
		private float[] _outputData;
		private HashSet<int> _missedOutputIndexes;

		public TrainPair() {
		}

		public TrainPair(float[] inputData, float[] outputData, 
				HashSet<int> missedInputIndexes = null, 
				HashSet<int> missedOutputIndexes = null) 
					: base(inputData, missedInputIndexes) {
			_outputData = outputData;
			_missedOutputIndexes = missedOutputIndexes;
		}

		public TrainPair(TrainPair source) : base(source) {
			var sourceOutputData = source._outputData;
			_outputData = new float[sourceOutputData.Length];
			sourceOutputData.CopyTo(_outputData, 0);

			_missedOutputIndexes = null;
			if (source._missedOutputIndexes != null) {
				_missedOutputIndexes = new HashSet<int>(source._missedOutputIndexes);
			}
		}

		public float[] Output {
			get { return _outputData; }
			set { _outputData = value; }
		}

		public int OutputLength {
			get { return _outputData.Length; }
		}

		public HashSet<int> MissedOutputIndexes {
			get { return _missedOutputIndexes; }
			set { _missedOutputIndexes = value; }
		}

		public new TrainPair Copy() {
			return new TrainPair(this);
		}
	}
}
