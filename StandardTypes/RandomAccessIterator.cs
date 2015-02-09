using System;
using System.Collections.Generic;

namespace StandardTypes {
	public sealed class RandomAccessIterator<T> {
		private readonly int[] _positions;
		private readonly IList<T> _sourceList;
		private readonly Random _random;
		private int _lastRandomAccessIndex;

		public RandomAccessIterator(IList<T> list) {
			_random = new Random();
			_sourceList = list;
			_positions = CreateStartPositions(_sourceList.Count);
			RefreshRandomAccess();
		}

		private static int[] CreateStartPositions(int size) {
			var positions = new int[size];
			for (var i = 0; i < size; i++) {
				positions[i] = i;
			}
			return positions;
		}

		public void RefreshRandomAccess() {
			for (var i = _sourceList.Count - 1; i > 0; i--) {
				var newIndex = _random.Next(i + 1);
				if (newIndex != i) {
					var tmp = _positions[newIndex];
					_positions[newIndex] = _positions[i];
					_positions[i] = tmp;
				}
			}

			_lastRandomAccessIndex = 0;
		}

		public T Next() {
			if (_lastRandomAccessIndex >= _sourceList.Count) {
				RefreshRandomAccess();
			}
			var result = _sourceList[_positions[_lastRandomAccessIndex]];
			_lastRandomAccessIndex++;
			return result;
		}

		public int Size() {
			return _sourceList.Count;
		}

		public IList<T> Collection {
			get { return _sourceList; }
		} 
	}
}