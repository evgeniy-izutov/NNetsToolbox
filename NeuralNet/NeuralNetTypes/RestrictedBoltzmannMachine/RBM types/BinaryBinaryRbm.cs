using System;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class BinaryBinaryRbm : RestrictedBoltzmannMachine {
		public BinaryBinaryRbm(int visibleStatesCount, int hiddenStatesCount) : base(visibleStatesCount, hiddenStatesCount) {
		}
		
		public override void VisibleLayerCalculateActivity() {
			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = visibleStatesBias[i];
			}

			for (var j = 0; j < hiddenStates.Length; j++) {
				var weightsStartPos = j*visibleStates.Length;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < visibleStates.Length; i++) {
					visibleStates[i] += hiddenState*weights[weightsStartPos + i];
				}
			}

			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = 1.0f/(1.0f + (float) Math.Exp(-visibleStates[i]));
			}
		}

		public override void HiddenLayerCalculateActivity() {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j];
				var weightsStartPos = j*visibleStates.Length;
				for (var i = 0; i < visibleStates.Length; i++) {
					sum += visibleStates[i]*weights[weightsStartPos + i];
				}
				hiddenStates[j] = 1.0f/(1.0f + (float) Math.Exp(-sum));
			}
		}

		public override void HiddenLayerCalculateActivity(float[] newVisibleState) {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j];
				var weightsStartPos = j*newVisibleState.Length;
				for (var i = 0; i < newVisibleState.Length; i++) {
					sum += newVisibleState[i]*weights[weightsStartPos + i];
				}
				hiddenStates[j] = 1.0f/(1.0f + (float) Math.Exp(-sum));
			}
		}

		public override void VisibleLayerCalculateActivity(float[] addedWeight, float[] addedVisibleBias) {
			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = visibleStatesBias[i] + addedVisibleBias[i];
			}

			for (var j = 0; j < hiddenStates.Length; j++) {
				var weightsStartPos = j*visibleStates.Length;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < visibleStates.Length; i++) {
					visibleStates[i] += hiddenState*(weights[weightsStartPos + i] + addedWeight[weightsStartPos + i]);
				}
			}

			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = 1.0f/(1.0f + (float) Math.Exp(-visibleStates[i]));
			}
		}

		public override void HiddenLayerCalculateActivity(float[] addedWeight, float[] addedHiddenBias) {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j] + addedHiddenBias[j];
				var weightsStartPos = j*visibleStates.Length;
				for (var i = 0; i < visibleStates.Length; i++) {
					sum += visibleStates[i]*(weights[weightsStartPos + i] + addedWeight[weightsStartPos + i]);
				}
				hiddenStates[j] = 1.0f/(1.0f + (float) Math.Exp(-sum));
			}
		}

		public override void HiddenLayerCalculateActivity(float[] newVisibleState, float[] addedWeight, float[] addedHiddenBias) {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j] + addedHiddenBias[j];
				var weightsStartPos = j*newVisibleState.Length;
				for (var i = 0; i < newVisibleState.Length; i++) {
					sum += newVisibleState[i]*(weights[weightsStartPos + i] + addedWeight[weightsStartPos + i]);
				}
				hiddenStates[j] = 1.0f/(1.0f + (float) Math.Exp(-sum));
			}
		}
	}
}