using System;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class EnhancedGradient : IGradientFunction {
		private readonly int _visibleStatesCount;
		private readonly int _hiddenStatesCount;
		private readonly float[] _dataVisibleHidden;
		private readonly float[] _dataVisible;
		private readonly float[] _dataHidden;
		private readonly float[] _modelVisibleHidden;
		private readonly float[] _modelVisible;
		private readonly float[] _modelHidden;

		public EnhancedGradient(int visibleStatesCount, int hiddenStatesCount) {
			_visibleStatesCount = visibleStatesCount;
			_hiddenStatesCount = hiddenStatesCount;

			_dataVisibleHidden = new float[_visibleStatesCount*_hiddenStatesCount];
			_modelVisibleHidden = new float[_visibleStatesCount*_hiddenStatesCount];
			Array.Clear(_dataVisibleHidden, 0, _visibleStatesCount*_hiddenStatesCount);
			Array.Clear(_modelVisibleHidden, 0, _visibleStatesCount*_hiddenStatesCount);
			
			_dataVisible = new float[_visibleStatesCount];
			_modelVisible = new float[_visibleStatesCount];
			Array.Clear(_dataVisible, 0, _visibleStatesCount);
			Array.Clear(_modelVisible, 0, _visibleStatesCount);
			
			_dataHidden = new float[_hiddenStatesCount];
			_modelHidden = new float[_hiddenStatesCount];
			Array.Clear(_dataHidden, 0, hiddenStatesCount);
			Array.Clear(_modelHidden, 0, hiddenStatesCount);
		}

		public void StorePositivePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var startIndex = j*_visibleStatesCount;
				var hiddenSate = hiddenStates[j];
				for (var i = 0; i < _visibleStatesCount; i++) {
					_dataVisibleHidden[startIndex + i] += visibleStates[i]*hiddenSate;
				}
				_dataHidden[j] += hiddenSate;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				_dataVisible[i] += visibleStates[i];
			}
		}

		public void StoreNegativePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var startIndex = j*_visibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < _visibleStatesCount; i++) {
					_modelVisibleHidden[startIndex + i] += visibleStates[i]*hiddenState;
				}
				_modelHidden[j] += hiddenState;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				_modelVisible[i] += visibleStates[i];
			}
		}

		public void MakeGradient(RbmGradients gradients, float packageFactor) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var dataHidden = _dataHidden[j];
				var modelHidden = _modelHidden[j];
				var hiddenStateSumGradient = 0f;
				for (var i = 0; i < _visibleStatesCount; i++) {
					var index = j*_visibleStatesCount + i;
					var weightGradient = packageFactor*(_dataVisibleHidden[index] - packageFactor*_dataVisible[i]*dataHidden -
						_modelVisibleHidden[index] + packageFactor*_modelVisible[i]*modelHidden);
					hiddenStateSumGradient += 0.5f*packageFactor*(_dataVisible[i] + _modelVisible[i])*weightGradient;
					gradients.PackageDerivativeForWeights[index] = weightGradient;
					_dataVisibleHidden[index] = 0f;
					_modelVisibleHidden[index] = 0f;
				}
				gradients.PackageDerivativeForHiddenBias[j] = packageFactor*(_dataHidden[j] - _modelHidden[j]) - hiddenStateSumGradient;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				var dif = 0f;
				for (var j = 0; j < _hiddenStatesCount; j++) {
					dif += 0.5f*packageFactor*(_dataHidden[j] + _modelHidden[j])*gradients.PackageDerivativeForWeights[j*_visibleStatesCount + i];
				}
				gradients.PackageDerivativeForVisibleBias[i] = packageFactor*(_dataVisible[i] - _modelVisible[i]) - dif;
				_dataVisible[i] = 0f;
				_modelVisible[i] = 0f;
			}

			Array.Clear(_dataHidden, 0, _dataHidden.Length);
			Array.Clear(_modelHidden, 0, _modelHidden.Length);
		}
	}
}