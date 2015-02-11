namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class LinearGradient : GradientFunction {
		public override void PrepareToNextPackage(int nextPackageSize) {}

		public override void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenSate = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					Gradients.PackageDerivativeForWeights[startIndex + i] += visibleStates[i]*hiddenSate;
				}
				Gradients.PackageDerivativeForHiddenBias[j] += hiddenSate;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				Gradients.PackageDerivativeForVisibleBias[i] += visibleStates[i];
			}
		}

		public override void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					Gradients.PackageDerivativeForWeights[startIndex + i] -= visibleStates[i]*hiddenState;
				}
				Gradients.PackageDerivativeForHiddenBias[j] -= hiddenState;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				Gradients.PackageDerivativeForVisibleBias[i] -= visibleStates[i];
			}
		}

		public override void MakeGradient(float packageFactor) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				for (var i = 0; i < VisibleStatesCount; i++) {
					Gradients.PackageDerivativeForWeights[startIndex + i] *= packageFactor;
				}
				Gradients.PackageDerivativeForHiddenBias[j] *= packageFactor;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				Gradients.PackageDerivativeForVisibleBias[i] *= packageFactor;
			}
		}
	}
}
