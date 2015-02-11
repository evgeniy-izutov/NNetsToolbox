using System;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class EnhancedGradient : GradientFunction {
		private float[] _dataVisibleHidden;
		private float[] _dataVisible;
		private float[] _dataHidden;
		private float[] _modelVisibleHidden;
		private float[] _modelVisible;
		private float[] _modelHidden;

		public override void PrepareToNextPackage(int nextPackageSize) {}

		public override void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_dataVisibleHidden[startIndex + i] += visibleStates[i]*hiddenState;
				}
				_dataHidden[j] += hiddenState;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_dataVisible[i] += visibleStates[i];
			}
		}

		public override void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_modelVisibleHidden[startIndex + i] += visibleStates[i]*hiddenState;
				}
				_modelHidden[j] += hiddenState;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_modelVisible[i] += visibleStates[i];
			}
		}

		public override void MakeGradient(float packageFactor) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var dataHidden = _dataHidden[j];
				var modelHidden = _modelHidden[j];
				var hiddenStateSumGradient = 0f;
				for (var i = 0; i < VisibleStatesCount; i++) {
					var index = j*VisibleStatesCount + i;
					var weightGradient = packageFactor*(_dataVisibleHidden[index] - packageFactor*_dataVisible[i]*dataHidden -
						_modelVisibleHidden[index] + packageFactor*_modelVisible[i]*modelHidden);
					hiddenStateSumGradient += 0.5f*packageFactor*(_dataVisible[i] + _modelVisible[i])*weightGradient;
					Gradients.PackageDerivativeForWeights[index] = weightGradient;
					_dataVisibleHidden[index] = 0f;
					_modelVisibleHidden[index] = 0f;
				}
				Gradients.PackageDerivativeForHiddenBias[j] = packageFactor*(_dataHidden[j] - _modelHidden[j]) - hiddenStateSumGradient;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				var dif = 0f;
				for (var j = 0; j < HiddenStatesCount; j++) {
					dif += 0.5f*packageFactor*(_dataHidden[j] + _modelHidden[j])*Gradients.PackageDerivativeForWeights[j*VisibleStatesCount + i];
				}
				Gradients.PackageDerivativeForVisibleBias[i] = packageFactor*(_dataVisible[i] - _modelVisible[i]) - dif;
				_dataVisible[i] = 0f;
				_modelVisible[i] = 0f;
			}

			Array.Clear(_dataHidden, 0, _dataHidden.Length);
			Array.Clear(_modelHidden, 0, _modelHidden.Length);
		}

		protected override void AllocateMemory() {
		    _dataVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			_modelVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			Array.Clear(_dataVisibleHidden, 0, VisibleStatesCount*HiddenStatesCount);
			Array.Clear(_modelVisibleHidden, 0, VisibleStatesCount*HiddenStatesCount);
			
			_dataVisible = new float[VisibleStatesCount];
			_modelVisible = new float[VisibleStatesCount];
			Array.Clear(_dataVisible, 0, VisibleStatesCount);
			Array.Clear(_modelVisible, 0, VisibleStatesCount);
			
			_dataHidden = new float[HiddenStatesCount];
			_modelHidden = new float[HiddenStatesCount];
			Array.Clear(_dataHidden, 0, HiddenStatesCount);
			Array.Clear(_modelHidden, 0, HiddenStatesCount);
		}
	}
}