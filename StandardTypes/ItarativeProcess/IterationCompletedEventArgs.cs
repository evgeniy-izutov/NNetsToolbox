using System;

namespace StandardTypes {
	public sealed class IterationCompletedEventArgs : EventArgs {
		private readonly int _iterationNum;
		private readonly float _iterationValue;
        private readonly float _addedIterationValue;

		public IterationCompletedEventArgs(int iterationNum, float iterationValue, float addedIterationValue) {
			_iterationNum = iterationNum;
			_iterationValue = iterationValue;
		    _addedIterationValue = addedIterationValue;
		}

		public int IterationNum {
			get { return _iterationNum; }
		}

		public float IterationValue {
			get { return _iterationValue; }
		}

        public float AddedIterationValue {
            get { return _addedIterationValue; }
        }
	}
}