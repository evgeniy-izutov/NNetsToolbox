using System;

namespace StandardTypes {
	public sealed class IterativeProcessFinishedEventArgs : EventArgs {
		private readonly int _iterationCount;

		public IterativeProcessFinishedEventArgs(int iterationCount) {
			_iterationCount = iterationCount;
		}

		public int IterationCount {
			get { return _iterationCount; }
		}
	}
}