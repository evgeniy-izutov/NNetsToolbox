using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.GenerativeRbm {
	public class FastPersistentContrastiveDivergence : RbmTrainMethod {
		private readonly float _fastWeightsDecreaseFactor;
		private float[] _fastWeights;
		private float[] _fastWeightsForVisibleBias;
		private float[] _fastWeightsForHiddenBias;
		private float[] _oldDeltaRegularWeights;
		private float[] _oldDeltaRegularWeightsForVisibleBias;
		private float[] _oldDeltaRegularWeightsForHiddenBias;
		protected float[][] persistentVisibleStates;

		public FastPersistentContrastiveDivergence(IList<TrainSingle> trainData, IGradientFunction gradient, float fastWeightsDecreaseFactor) : base(trainData, gradient) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
		}

		public FastPersistentContrastiveDivergence(IList<TrainSingle> trainData, IList<TrainSingle> testData, IGradientFunction gradient, float fastWeightsDecreaseFactor) : base(trainData, testData, gradient) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
		}

		protected override void AllocateMemory() {
			var weightsCount = neuralNet.Weights.Length;
			_oldDeltaRegularWeights = new float[weightsCount];
			_fastWeights = new float[weightsCount];
			for (var i = 0; i < weightsCount; i++) {
				_oldDeltaRegularWeights[i] = 0f;
				_fastWeights[i] = 0f;
			}

			_oldDeltaRegularWeightsForVisibleBias = new float[visibleStatesCount];
			_fastWeightsForVisibleBias = new float[visibleStatesCount];
			for (var i = 0; i < visibleStatesCount; i++) {
				_oldDeltaRegularWeightsForVisibleBias[i] = 0f;
				_fastWeightsForVisibleBias[i] = 0f;
			}

			persistentVisibleStates = new float[packagesCount][];
			for (var i = 0; i < packagesCount; i++) {
				persistentVisibleStates[i] = new float[visibleStatesCount];
				for (var j = 0; j < visibleStatesCount; j++) {
					persistentVisibleStates[i][j] = 0f;
				}
			}

			_oldDeltaRegularWeightsForHiddenBias = new float[hiddenStatesCount];
			_fastWeightsForHiddenBias = new float[hiddenStatesCount];
			for (var i = 0; i < hiddenStatesCount; i++) {
				_oldDeltaRegularWeightsForHiddenBias[i] = 0f;
				_fastWeightsForHiddenBias[i] = 0f;
			}
		}

		protected override void ClearReference() {
			persistentVisibleStates = null;
			_fastWeights = null;
			_fastWeightsForVisibleBias = null;
			_fastWeightsForHiddenBias = null;
			_oldDeltaRegularWeights = null;
			_oldDeltaRegularWeightsForVisibleBias = null;
			_oldDeltaRegularWeightsForHiddenBias = null;
		}

		protected override void MakePositivePhase(float[] input) {
			neuralNet.HiddenLayerCalculateActivity(input);
		}

		protected override void MakeNegativePhase(int packageId) {
			neuralNet.HiddenLayerCalculateActivity(persistentVisibleStates[packageId], _fastWeights, _fastWeightsForHiddenBias);
		}

		protected override float[] GetVisibleStatesOnNegativePhase(int packageId) {
			return persistentVisibleStates[packageId];
		}

		protected override float[] GetHiddenStatesOnNegativePhase() {
			return neuralNet.HiddenStates;
		}

		protected override void RestoreVisibleStates(int packageId) {
			neuralNet.VisibleLayerCalculateActivity(_fastWeights, _fastWeightsForVisibleBias);
			neuralNet.CopyVisibleLayerTo(persistentVisibleStates[packageId]);
		}

		protected override void ModifyWeightsOfNeuronNet() {
			var curRegularLearnSpeed = properties.BaseLearnSpeed*properties.LearnFactorStrategy.GetFactor(epochNumber);
			var curFastLearnSpeed = properties.BaseLearnSpeed*properties.AddedLearnFactorStrategy.GetFactor(epochNumber);

			var regularWeights = neuralNet.Weights;
			for (var j = 0; j < hiddenStatesCount; j++) {
				var startIndex = j*visibleStatesCount;
				for (var i = 0; i < visibleStatesCount; i++) {
					var weightIndex = startIndex + i;

					var partialDerivative = gradients.PackageDerivativeForWeights[weightIndex];
					gradients.PackageDerivativeForWeights[weightIndex] = 0.0f;
					
					var newDeltaRegularWeight = curRegularLearnSpeed*(partialDerivative +
                        properties.Momentum*_oldDeltaRegularWeights[weightIndex] - 
						properties.Regularization.GetDerivative(regularWeights[weightIndex]));
					_oldDeltaRegularWeights[weightIndex] = newDeltaRegularWeight;
					regularWeights[weightIndex] += (1f + properties.Momentum)*newDeltaRegularWeight;

					_fastWeights[weightIndex] = _fastWeightsDecreaseFactor*_fastWeights[weightIndex] +
                                                curFastLearnSpeed*partialDerivative;
				}
			}

			var regularVisibleStatesBias = neuralNet.VisibleStatesBias;
			for (var i = 0; i < visibleStatesCount; i++) {
				var partialDerivativeForVisibleBias = gradients.PackageDerivativeForVisibleBias[i];
				gradients.PackageDerivativeForVisibleBias[i] = 0f;
				
				var newDeltaForRegularVisibleBias = curRegularLearnSpeed*partialDerivativeForVisibleBias +
					                                properties.Momentum*_oldDeltaRegularWeightsForVisibleBias[i];
				_oldDeltaRegularWeightsForVisibleBias[i] = newDeltaForRegularVisibleBias;
				regularVisibleStatesBias[i] += (1f + properties.Momentum)*newDeltaForRegularVisibleBias;

				_fastWeightsForVisibleBias[i] = _fastWeightsDecreaseFactor*_fastWeightsForVisibleBias[i] +
					curFastLearnSpeed*partialDerivativeForVisibleBias;
			}

			var regularHiddenStatesBias = neuralNet.HiddenStatesBias;
			for (var j = 0; j < hiddenStatesCount; j++) {
				var partialDerivativeForHiddenBias = gradients.PackageDerivativeForHiddenBias[j];
				gradients.PackageDerivativeForHiddenBias[j] = 0f;
				
				var newDeltaForRegularHiddenBias = curRegularLearnSpeed*partialDerivativeForHiddenBias +
					                               properties.Momentum*_oldDeltaRegularWeightsForHiddenBias[j];
				_oldDeltaRegularWeightsForHiddenBias[j] = newDeltaForRegularHiddenBias;
				regularHiddenStatesBias[j] += (1f + properties.Momentum)*newDeltaForRegularHiddenBias;

				_fastWeightsForHiddenBias[j] = _fastWeightsDecreaseFactor*_fastWeightsForHiddenBias[j] +
					curFastLearnSpeed*partialDerivativeForHiddenBias;
			}
		}
	}
}
