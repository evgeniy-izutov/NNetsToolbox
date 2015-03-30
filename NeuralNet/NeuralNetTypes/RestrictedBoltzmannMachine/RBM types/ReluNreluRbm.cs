using System;
using MathNet.Numerics.Distributions;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class ReluNreluRbm : RestrictedBoltzmannMachine {
		private readonly Normal _normalGenerator;

		public ReluNreluRbm() : base() {
			_normalGenerator = new Normal {
				RandomSource = new Random()
			};
		}
		
		public ReluNreluRbm(int visibleStatesCount, int hiddenStatesCount) : base(visibleStatesCount, hiddenStatesCount) {
			_normalGenerator = new Normal {
				RandomSource = new Random()
			};
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
		}

		public override void HiddenLayerCalculateActivity() {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j];
				var weightsStartPos = j*visibleStates.Length;
				for (var i = 0; i < visibleStates.Length; i++) {
					sum += visibleStates[i]*weights[weightsStartPos + i];
				}
				var sigma = (float) (1.0/Math.Sqrt(1.0 + Math.Exp(-sum)));
				hiddenStates[j] = Math.Max(0.0f, sum + ((float) _normalGenerator.Sample())*sigma);
			}
		}

		public override void HiddenLayerCalculateActivity(float[] newVisibleState) {
			for (var j = 0; j < hiddenStates.Length; j++) {
				var sum = hiddenStatesBias[j];
				var weightsStartPos = j*newVisibleState.Length;
				for (var i = 0; i < newVisibleState.Length; i++) {
					sum += newVisibleState[i]*weights[weightsStartPos + i];
				}
				var sigma = (float) (1.0/Math.Sqrt(1.0 + Math.Exp(-sum)));
				hiddenStates[j] = Math.Max(0.0f, sum + ((float) _normalGenerator.Sample())*sigma);
			}
		}

		public override void VisibleLayerCalculateActivity(float[] addedWeight, float[] addedVisibleBias) {
			throw new NotImplementedException();
		}

		public override void HiddenLayerCalculateActivity(float[] addedWeight, float[] addedHiddenBias) {
			throw new NotImplementedException();
		}

		public override void HiddenLayerCalculateActivity(float[] newVisibleState, float[] addedWeight, float[] addedHiddenBias) {
			throw new NotImplementedException();
		}

		public override void VisibleLayerSampling() {
			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = Math.Max(0.0f, visibleStates[i] + ((float) _normalGenerator.Sample()));
			}
		}

		public override void HiddenLayerSampling() {
		}
	}
}