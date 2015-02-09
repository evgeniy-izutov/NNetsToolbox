namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class LinearGradient : IGradientFunction {
		private readonly int _visibleStatesCount;
		private readonly int _hiddenStatesCount;

		public LinearGradient(int visibleStatesCount, int hiddenStatesCount) {
			_visibleStatesCount = visibleStatesCount;
			_hiddenStatesCount = hiddenStatesCount;
		}

		public void StorePositivePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var startIndex = j*_visibleStatesCount;
				var hiddenSate = hiddenStates[j];
				for (var i = 0; i < _visibleStatesCount; i++) {
					gradients.PackageDerivativeForWeights[startIndex + i] += visibleStates[i]*hiddenSate;
				}
				gradients.PackageDerivativeForHiddenBias[j] += hiddenSate;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				gradients.PackageDerivativeForVisibleBias[i] += visibleStates[i];
			}
		}

		public void StoreNegativePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var startIndex = j*_visibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < _visibleStatesCount; i++) {
					gradients.PackageDerivativeForWeights[startIndex + i] -= visibleStates[i]*hiddenState;
				}
				gradients.PackageDerivativeForHiddenBias[j] -= hiddenState;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				gradients.PackageDerivativeForVisibleBias[i] -= visibleStates[i];
			}
		}

		public void MakeGradient(RbmGradients gradients, float packageFactor) {
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var startIndex = j*_visibleStatesCount;
				for (var i = 0; i < _visibleStatesCount; i++) {
					gradients.PackageDerivativeForWeights[startIndex + i] *= packageFactor;
				}
				gradients.PackageDerivativeForHiddenBias[j] *= packageFactor;
			}

			for (var i = 0; i < _visibleStatesCount; i++) {
				gradients.PackageDerivativeForVisibleBias[i] *= packageFactor;
			}
		}
	}
}
