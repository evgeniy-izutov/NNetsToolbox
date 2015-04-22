using System;
using System.Collections.Generic;

namespace StandardTypes {
	public sealed class CvDataSeparator<T> where T : ICopyType<T> {
		private readonly IList<T> _sourceList;
		private readonly int _cvPartsCount;
		private readonly int _cvPartSize;
		private int _cvIterNumber;

		public CvDataSeparator(IList<T> list, int cvPartsCount) {
			_sourceList = list;
			_cvPartsCount = cvPartsCount;
			_cvPartSize = list.Count/_cvPartsCount;
			_cvIterNumber = -1;
			MixData();
		}

		public void Refresh() {
			_cvIterNumber = -1;
			MixData();
		}

		public void Next(out List<T> trainData, out List<T> vTestData) {
			_cvIterNumber++;
			if (_cvIterNumber >= _cvPartsCount) {
				_cvIterNumber = 0;
			}

			trainData = new List<T>(_sourceList.Count - _cvPartSize);
			for (var i = 0; i < _cvIterNumber*_cvPartSize; i++) {
				trainData.Add(_sourceList[i].Copy());
			}
			for (var i = (_cvIterNumber + 1)*_cvPartSize; i < _sourceList.Count; i++) {
				trainData.Add(_sourceList[i].Copy());
			}

			vTestData = new List<T>(_cvPartSize);
			for (var i = _cvIterNumber*_cvPartSize; (i < (_cvIterNumber + 1)*_cvPartSize) && (i < _sourceList.Count); i++) {
				vTestData.Add(_sourceList[i].Copy());
			}
		}

		private void MixData() {
			var uniformGeberator = new Random();
			for (var i = _sourceList.Count - 1; i > 0; i--) {
				var newIndex = uniformGeberator.Next(i + 1);
				if (newIndex != i) {
					var tmp = _sourceList[newIndex];
					_sourceList[newIndex] = _sourceList[i];
					_sourceList[i] = tmp;
				}
			}
		}
	}
}
