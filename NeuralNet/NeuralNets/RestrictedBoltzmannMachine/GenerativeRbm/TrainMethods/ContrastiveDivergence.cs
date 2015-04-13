using System;
using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.GenerativeRbm {
	public sealed class ContrastiveDivergence : RbmTrainMethod {
		private readonly int _methodStepsCount;
		private float[] _oldDeltaWeights;
		private float[] _learnFactors;
		private float[] _derivativeAverages;
		private float[] _oldDeltaWeightsForVisibleBias;
		private float[] _learnFactorsForVisibleBias;
		private float[] _oldDeltaWeightsForHiddenBias;
		private float[] _learnFactorsForHiddenBias;
		private float[] _derivativeAveragesForVisibleBias;
		private float[] _derivativeAveragesForHiddenBias;

		public ContrastiveDivergence(IList<TrainSingle> trainData, IGradientFunction gradient, int methodStepsCount) : base(trainData, gradient) {
			_methodStepsCount = methodStepsCount;
		}

		public ContrastiveDivergence(IList<TrainSingle> trainData, IList<TrainSingle> testData, IGradientFunction gradient, int methodStepsCount) : base(trainData, testData, gradient) {
			_methodStepsCount = methodStepsCount;
		}

		protected override void AllocateMemory() {		
			var weightsCount = neuralNet.Weights.Length;
			_oldDeltaWeights = new float[weightsCount];
			_derivativeAverages = new float[weightsCount];
			_learnFactors = new float[weightsCount];
			for (var i = 0; i < weightsCount; i++) {
				_oldDeltaWeights[i] = 0.0f;
				_derivativeAverages[i] = 0.0f;
				_learnFactors[i] = 1.0f;
			}

			_oldDeltaWeightsForVisibleBias = new float[visibleStatesCount];
			_derivativeAveragesForVisibleBias = new float[visibleStatesCount];
			_learnFactorsForVisibleBias = new float[visibleStatesCount];
			for (var i = 0; i < visibleStatesCount; i++) {
				_oldDeltaWeightsForVisibleBias[i] = 0.0f;
				_derivativeAveragesForVisibleBias[i] = 0.0f;
				_learnFactorsForVisibleBias[i] = 1.0f;
			}

			_oldDeltaWeightsForHiddenBias = new float[hiddenStatesCount];
			_derivativeAveragesForHiddenBias = new float[hiddenStatesCount];
			_learnFactorsForHiddenBias = new float[hiddenStatesCount];
			for (var i = 0; i < hiddenStatesCount; i++) {
				_oldDeltaWeightsForHiddenBias[i] = 0.0f;
				_derivativeAveragesForHiddenBias[i] = 0.0f;
				_learnFactorsForHiddenBias[i] = 1.0f;
			}
		}

		protected override void ClearReference() {
			_oldDeltaWeights = null;
			_oldDeltaWeightsForVisibleBias = null;
			_oldDeltaWeightsForHiddenBias = null;
			_derivativeAverages = null;
			_derivativeAveragesForVisibleBias = null;
			_derivativeAveragesForHiddenBias = null;
			_learnFactors = null;
			_learnFactorsForVisibleBias = null;
			_learnFactorsForHiddenBias = null;
		}

		protected override void MakePositivePhase(float[] input) {
			neuralNet.HiddenLayerCalculateActivity(input);
		}

		protected override void MakeNegativePhase(int packageId) {
			neuralNet.HiddenLayerSampling();
			neuralNet.VisibleLayerCalculateActivity();
			for (var k = 1; k < _methodStepsCount; k++) {
				neuralNet.HiddenLayerCalculateActivity();
				neuralNet.HiddenLayerSampling();
				neuralNet.VisibleLayerCalculateActivity();
			}
			neuralNet.HiddenLayerCalculateActivity();
		}

		protected override float[] GetVisibleStatesOnNegativePhase(int packageId) {
			return neuralNet.VisibleStates;
		}

		protected override float[] GetHiddenStatesOnNegativePhase() {
			return neuralNet.HiddenStates;
		}

	    protected override void RestoreVisibleStates(int packageId) {
	    }

		protected override void ModifyWeightsOfNeuronNet() {
			var curLearnSpeed = properties.BaseLearnSpeed*properties.LearnFactorStrategy.GetFactor(epochNumber);
			var weights = neuralNet.Weights;
			for (var j = 0; j < hiddenStatesCount; j++) {
				for (var i = 0; i < visibleStatesCount; i++) {
					var weightIndex = j*visibleStatesCount + i;
					
					var lastDerivativeAverage = _derivativeAverages[weightIndex];
					var partialDerivative = gradients.PackageDerivativeForWeights[weightIndex] - 
						properties.Regularization.GetDerivative(weights[weightIndex]);
					gradients.PackageDerivativeForWeights[weightIndex] = 0.0f;
					_learnFactors[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
						Math.Min(_learnFactors[weightIndex] + properties.SpeedBonus, properties.SpeedUpBorder):
						Math.Max(_learnFactors[weightIndex]*properties.SpeedPenalty, properties.SpeedLowBorder);
					_derivativeAverages[weightIndex] = properties.AverageLearnFactor*partialDerivative + 
						(1.0f - properties.AverageLearnFactor)*lastDerivativeAverage;

					var newDeltaWeight = curLearnSpeed*_learnFactors[weightIndex]*partialDerivative + 
                                         properties.Momentum*_oldDeltaWeights[weightIndex];
					_oldDeltaWeights[weightIndex] = newDeltaWeight;
					weights[weightIndex] += (1f + properties.Momentum)*newDeltaWeight;
				}
			}

			var visibleStatesBias = neuralNet.VisibleStatesBias;
			for (var i = 0; i < visibleStatesCount; i++) {
				var lastDerivativeAverageForVisibleBias = _derivativeAveragesForVisibleBias[i];
				var partialDerivativeForVisibleBias = gradients.PackageDerivativeForVisibleBias[i];
				gradients.PackageDerivativeForVisibleBias[i] = 0.0f;
				_learnFactorsForVisibleBias[i] = (lastDerivativeAverageForVisibleBias*partialDerivativeForVisibleBias > 0.0f) ?
					Math.Min(_learnFactorsForVisibleBias[i] + properties.SpeedBonus, properties.SpeedUpBorder):
					Math.Max(_learnFactorsForVisibleBias[i]*properties.SpeedPenalty, properties.SpeedLowBorder);
				_derivativeAveragesForVisibleBias[i] = properties.AverageLearnFactor*partialDerivativeForVisibleBias + 
					(1.0f - properties.AverageLearnFactor)*lastDerivativeAverageForVisibleBias;

				var newDeltaForVisibleBias = curLearnSpeed*_learnFactorsForVisibleBias[i]*partialDerivativeForVisibleBias + 
					                         properties.Momentum*_oldDeltaWeightsForVisibleBias[i];
				_oldDeltaWeightsForVisibleBias[i] = newDeltaForVisibleBias;
				visibleStatesBias[i] += (1f + properties.Momentum)*newDeltaForVisibleBias;
			}

			var hiddenStatesBias = neuralNet.HiddenStatesBias;
			for (var j = 0; j < hiddenStatesCount; j++) {
				var lastDerivativeAverageForHiddenBias = _derivativeAveragesForHiddenBias[j];
				var partialDerivativeForHiddenBias = gradients.PackageDerivativeForHiddenBias[j];
				gradients.PackageDerivativeForHiddenBias[j] = 0.0f;
				_learnFactorsForHiddenBias[j] = (lastDerivativeAverageForHiddenBias*partialDerivativeForHiddenBias > 0.0f) ?
					Math.Min(_learnFactorsForHiddenBias[j] + properties.SpeedBonus, properties.SpeedUpBorder):
					Math.Max(_learnFactorsForHiddenBias[j]*properties.SpeedPenalty, properties.SpeedLowBorder);
				_derivativeAveragesForHiddenBias[j] = properties.AverageLearnFactor*partialDerivativeForHiddenBias + 
					(1.0f - properties.AverageLearnFactor)*lastDerivativeAverageForHiddenBias;

				var newDeltaForHiddenBias = curLearnSpeed*_learnFactorsForHiddenBias[j]*partialDerivativeForHiddenBias + 
					                        properties.Momentum*_oldDeltaWeightsForHiddenBias[j];
				_oldDeltaWeightsForHiddenBias[j] = newDeltaForHiddenBias;
				hiddenStatesBias[j] += (1f + properties.Momentum)*newDeltaForHiddenBias;
			}
		}
	}
}
