using System;
using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.ClassificationRbm {
	public sealed class DiscriminativeTrainMethod : TrainMethod {
		private readonly RandomAccessIterator<TrainPair> _trainDataIterator;
		private readonly IList<TrainPair> _testData;
		private float[] _labelsPrediction;
        private float _packageFactor;
		private ITrainProperties _properties;
		private ClassificationRbm _neuralNet;
		private int _visibleStatesCount;
		private int _hiddenStatesCount;
		private int _labelsCount;
		private int _epochNumber;
		private int _packagesCount;

		private float[] _packageDerivativeForVisibleWeights;
		private float[] _packageDerivativeForLabelWeights;
		private float[] _packageDerivativeForHiddenBias;
		private float[] _packageDerivativeForLabelBias;
		
		private float[] _oldDeltaVisibleWeights;
		private float[] _oldDeltaLabelWeights;
		private float[] _learnFactorsVisibleWeights;
		private float[] _learnFactorsLabelWeights;
		private float[] _derivativeAveragesVisibleWeights;
		private float[] _derivativeAveragesLabelWeights;
		private float[] _oldDeltaWeightsForHiddenBias;
		private float[] _learnFactorsForHiddenBias;
		private float[] _derivativeAveragesForHiddenBias;
		private float[] _oldDeltaWeightsForLabelBias;
		private float[] _learnFactorsForLabelBias;
		private float[] _derivativeAveragesForLabelBias;

		private double[] _unweightedPrediction;
		private float[] _predictions;
		private float[] _activationSums;
		private float[] _sigmaValues;

		public DiscriminativeTrainMethod(IList<TrainPair> trainData) {
			_trainDataIterator = new RandomAccessIterator<TrainPair>(trainData);
		}

		public DiscriminativeTrainMethod(IList<TrainPair> trainData, IList<TrainPair> testData) {
			_trainDataIterator = new RandomAccessIterator<TrainPair>(trainData);
			_testData = testData;
		}
		
		public override void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties) {
			if (!(neuralNet is ClassificationRbm)) {
				throw new ArgumentException("Neural net has other structure");
			}

			_neuralNet = (ClassificationRbm) neuralNet;
			_visibleStatesCount = _neuralNet.VisibleStates.Length;
			_hiddenStatesCount = _neuralNet.HiddenStates.Length;
			_labelsCount = _neuralNet.Labels.Length;

			_labelsPrediction = new float[_labelsCount];

			_properties = trainProperties;
			_packageFactor = 1f/_properties.PackageSize;
			_packagesCount = CalculatePackagesCount();

			AllocateMemory();

			ProcessSate = IterativeProcessState.NotStarted;
		}

		protected override void ApplyResults() {
			if (ProcessSate == IterativeProcessState.Finished) {
				ClearReference();
			}
		}

		public override ITrainProperties Properties {
			get { return _properties; }
		}

		private void AllocateMemory() {
			var visibleWeightsCount = _neuralNet.VisibleStatesWeights.Length;
			_packageDerivativeForVisibleWeights = new float[visibleWeightsCount];
			_oldDeltaVisibleWeights = new float[visibleWeightsCount];
			_derivativeAveragesVisibleWeights = new float[visibleWeightsCount];
			_learnFactorsVisibleWeights = new float[visibleWeightsCount];
			for (var i = 0; i < visibleWeightsCount; i++) {
				_packageDerivativeForVisibleWeights[i] = 0.0f;
				_oldDeltaVisibleWeights[i] = 0.0f;
				_derivativeAveragesVisibleWeights[i] = 0.0f;
				_learnFactorsVisibleWeights[i] = 1.0f;
			}

			var labelWeightsCount = _neuralNet.LabelsWeights.Length;
			_packageDerivativeForLabelWeights = new float[labelWeightsCount];
			_oldDeltaLabelWeights = new float[labelWeightsCount];
			_derivativeAveragesLabelWeights = new float[labelWeightsCount];
			_learnFactorsLabelWeights = new float[labelWeightsCount];
			for (var i = 0; i < labelWeightsCount; i++) {
				_packageDerivativeForLabelWeights[i] = 0.0f;
				_oldDeltaLabelWeights[i] = 0.0f;
				_derivativeAveragesLabelWeights[i] = 0.0f;
				_learnFactorsLabelWeights[i] = 1.0f;
			}

			_packageDerivativeForHiddenBias = new float[_hiddenStatesCount];
			_oldDeltaWeightsForHiddenBias = new float[_hiddenStatesCount];
			_derivativeAveragesForHiddenBias = new float[_hiddenStatesCount];
			_learnFactorsForHiddenBias = new float[_hiddenStatesCount];
			for (var i = 0; i < _hiddenStatesCount; i++) {
				_packageDerivativeForHiddenBias[i] = 0.0f;
				_oldDeltaWeightsForHiddenBias[i] = 0.0f;
				_derivativeAveragesForHiddenBias[i] = 0.0f;
				_learnFactorsForHiddenBias[i] = 1.0f;
			}

			_packageDerivativeForLabelBias = new float[_labelsCount];
			_oldDeltaWeightsForLabelBias = new float[_labelsCount];
			_derivativeAveragesForLabelBias = new float[_labelsCount];
			_learnFactorsForLabelBias = new float[_labelsCount];
			for (var i = 0; i < _labelsCount; i++) {
				_packageDerivativeForLabelBias[i] = 0.0f;
				_oldDeltaWeightsForLabelBias[i] = 0.0f;
				_derivativeAveragesForLabelBias[i] = 0.0f;
				_learnFactorsForLabelBias[i] = 1.0f;
			}

			_activationSums = new float[_hiddenStatesCount];
			_predictions = new float[_labelsCount];
			_unweightedPrediction = new double[_labelsCount];
			_sigmaValues = new float[_labelsCount];
		}

		private void ClearReference() {
			_oldDeltaVisibleWeights = null;
			_oldDeltaLabelWeights = null;
			_oldDeltaWeightsForHiddenBias = null;
			_oldDeltaWeightsForLabelBias = null;
			_derivativeAveragesVisibleWeights = null;
			_derivativeAveragesLabelWeights = null;
			_derivativeAveragesForHiddenBias = null;
			_derivativeAveragesForLabelBias = null;
			_learnFactorsVisibleWeights = null;
			_learnFactorsLabelWeights = null;
			_learnFactorsForHiddenBias = null;
			_learnFactorsForLabelBias = null;

			_packageDerivativeForVisibleWeights = null;
			_packageDerivativeForLabelWeights = null;
			_packageDerivativeForHiddenBias = null;
			_packageDerivativeForLabelBias = null;

			_activationSums = null;
			_predictions = null;
			_unweightedPrediction = null;
			_sigmaValues = null;
		}

		protected override void RunIterativeProcess() {
			if (IsTestDataAvailable()) {
	            RunTraingWithTesting();
	        }
			else {
				RunTraingWithoutTesting();
			}
		}

		private bool IsTestDataAvailable() {
			return !(_testData == null || _testData.Count == 0);
		}

		private void RunTraingWithTesting() {
			var trainError = TestModel(_trainDataIterator.Collection);
			var slidingTestError = TestModel(_testData);
			var minSlidingTestError = slidingTestError;
			_epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) &&
				   (trainError > _properties.Epsilon) &&
				   (_epochNumber <= _properties.MaxIterationCount) &&
				   ((_epochNumber <= _properties.SkipCvLimitFirstIterations) ||
				    (Math.Abs(slidingTestError - minSlidingTestError) < _properties.CvLimit))) {
				
				TrainEpoch();

				trainError = TestModel(_trainDataIterator.Collection);
			    var testError = TestModel(_testData);
				slidingTestError = _properties.CvSlidingFactor*testError +
					(1f - _properties.CvSlidingFactor)*slidingTestError;
				
				if (slidingTestError < minSlidingTestError) {
					minSlidingTestError = slidingTestError;
				}

                OnIterationCompleted(new IterationCompletedEventArgs(_epochNumber, trainError, testError));
				_epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(_epochNumber));
		}

		private void RunTraingWithoutTesting() {
			var trainError = TestModel(_trainDataIterator.Collection);
			_epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) && 
				   (trainError > _properties.Epsilon) && 
				   (_epochNumber <= _properties.MaxIterationCount)) {
				
				TrainEpoch();

				trainError = TestModel(_trainDataIterator.Collection);

                OnIterationCompleted(new IterationCompletedEventArgs(_epochNumber, trainError, float.NaN));
				_epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(_epochNumber));
		}

		private int CalculatePackagesCount() {
			var count = _trainDataIterator.Size()/_properties.PackageSize;
			if (_trainDataIterator.Size()%_properties.PackageSize != 0) {
				count++;
			}
			return count;
		}

		private float TestModel(IList<TrainPair> data) {
			var sumError = 0.0f;
			for (var i = 0; i < data.Count; i++) {
				var testExample = data[i];
				_neuralNet.Predict(testExample.Input, _labelsPrediction);
				sumError += _properties.Metrics.Calculate(testExample.Output, _labelsPrediction);
			}
			return sumError / data.Count;
		}

		private void TrainEpoch() {
			_trainDataIterator.RefreshRandomAccess();
			for (var i = 0; i < _packagesCount; i++) {
				TrainPackage(i);
			}
		}

		private void TrainPackage(int packageId) {
			for (var i = 0; i < _properties.PackageSize; i++) {
				var curTrainPair = _trainDataIterator.Next();
				
				var input = curTrainPair.Input;
				var labels = curTrainPair.Output;
				
				StorePackageGradient(input, labels);
			}
			ModifyWeightsOfNeuronNet();
		}

		private void StorePackageGradient(float[] visibleStates, float[] labels) {
			var labelsWeights = _neuralNet.LabelsWeights;
			
			MakePrediction(visibleStates, _predictions, _activationSums);

			for (var j = 0; j < _hiddenStatesCount; j++) {
				var activationSum = _activationSums[j];

				var weightedSigmaSum = 0f;
				var startIndex = j*_labelsCount;
				for (var k = 0; k < _labelsCount; k++) {
					var sigma = 1f/(1f + (float) Math.Exp(-activationSum - labelsWeights[startIndex + k]));
					_sigmaValues[k] = sigma;
					weightedSigmaSum += sigma*(labels[k] - _predictions[k]);
				}

				_packageDerivativeForHiddenBias[j] += _packageFactor*weightedSigmaSum;

				startIndex = j*_visibleStatesCount;
				for (var i = 0; i < _visibleStatesCount; i++) {
					_packageDerivativeForVisibleWeights[startIndex + i] += _packageFactor*weightedSigmaSum*visibleStates[i];
				}
				
				startIndex = j*_labelsCount;
				for (var k = 0; k < _labelsCount; k++) {
					_packageDerivativeForLabelWeights[startIndex + k] += _packageFactor*_sigmaValues[k]*
						(labels[k] - _predictions[k]);
				}
			}

			for (var k = 0; k < _labelsCount; k++) {
				_packageDerivativeForLabelBias[k] += _packageFactor*(labels[k] - _predictions[k]);
			}
		}

		private void MakePrediction(float[] input, float[] predictions, float[] activationSums) {
			var hiddenStatesBias = _neuralNet.HiddenStatesBias;
			var labelsBias = _neuralNet.LabelsBias;
			var visibleStatesWeights = _neuralNet.VisibleStatesWeights;
			var labelsWeights = _neuralNet.LabelsWeights;
			
			float activationSum;
			for (var j = 0; j < _hiddenStatesCount; j++) {
				activationSum = hiddenStatesBias[j];
				
				var weightsStartPos = j*_visibleStatesCount;
				for (var k = 0; k < _visibleStatesCount; k++) {
					activationSum += input[k]*visibleStatesWeights[weightsStartPos + k];
				}

				activationSums[j] = activationSum;
			}

			for (var k = 0; k < _labelsCount; k++) {
				_unweightedPrediction[k] = (float) Math.Exp(labelsBias[k]);
			}

			for (var j = 0; j < _hiddenStatesCount; j++) {
				activationSum = activationSums[j];
				for (var k = 0; k < _labelsCount; k++) {
					_unweightedPrediction[k] *= 1d + Math.Exp(activationSum + labelsWeights[j*_labelsCount + k]);
				}
			}

			var predictionsSum = 0d;
			for (var k = 0; k < _labelsCount; k++) {
				predictionsSum += _unweightedPrediction[k];
			}

			for (var k = 0; k < _labelsCount; k++) {
				predictions[k] = (float) (_unweightedPrediction[k]/predictionsSum);
			}
		}

		private void ModifyWeightsOfNeuronNet() {
			var curLearnSpeed = _properties.BaseLearnSpeed*_properties.LearnFactorStrategy.GetFactor(_epochNumber);
			var visibleWeights = _neuralNet.VisibleStatesWeights;
			var labelWeights = _neuralNet.LabelsWeights;
			for (var j = 0; j < _hiddenStatesCount; j++) {
				for (var i = 0; i < _visibleStatesCount; i++) {
					var weightIndex = j*_visibleStatesCount + i;
					
					var lastDerivativeAverage = _derivativeAveragesVisibleWeights[weightIndex];
					var partialDerivative = _packageDerivativeForVisibleWeights[weightIndex] - 
						_properties.Regularization.GetDerivative(visibleWeights[weightIndex]);
					_packageDerivativeForVisibleWeights[weightIndex] = 0f;
					_learnFactorsVisibleWeights[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
						Math.Min(_learnFactorsVisibleWeights[weightIndex] + _properties.SpeedBonus, _properties.SpeedUpBorder):
						Math.Max(_learnFactorsVisibleWeights[weightIndex]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
					_derivativeAveragesVisibleWeights[weightIndex] = _properties.AverageLearnFactor*partialDerivative + 
						(1f - _properties.AverageLearnFactor)*lastDerivativeAverage;

					var newDeltaWeight = curLearnSpeed*_learnFactorsVisibleWeights[weightIndex]*partialDerivative + 
                                         _properties.Momentum*_oldDeltaVisibleWeights[weightIndex];
					_oldDeltaVisibleWeights[weightIndex] = newDeltaWeight;
					visibleWeights[weightIndex] += (1f + _properties.Momentum)*newDeltaWeight;
				}


				for (var k = 0; k < _labelsCount; k++) {
					var weightIndex = j*_labelsCount + k;
					
					var lastDerivativeAverage = _derivativeAveragesLabelWeights[weightIndex];
					var partialDerivative = _packageDerivativeForLabelWeights[weightIndex] - 
						_properties.Regularization.GetDerivative(labelWeights[weightIndex]);
					_packageDerivativeForLabelWeights[weightIndex] = 0f;
					_learnFactorsLabelWeights[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
						Math.Min(_learnFactorsLabelWeights[weightIndex] + _properties.SpeedBonus, _properties.SpeedUpBorder):
						Math.Max(_learnFactorsLabelWeights[weightIndex]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
					_derivativeAveragesLabelWeights[weightIndex] = _properties.AverageLearnFactor*partialDerivative + 
						(1f - _properties.AverageLearnFactor)*lastDerivativeAverage;

					var newDeltaWeight = curLearnSpeed*_learnFactorsLabelWeights[weightIndex]*partialDerivative + 
                                         _properties.Momentum*_oldDeltaLabelWeights[weightIndex];
					_oldDeltaLabelWeights[weightIndex] = newDeltaWeight;
					labelWeights[weightIndex] += (1f + _properties.Momentum)*newDeltaWeight;
				}
			}

			var hiddenStatesBias = _neuralNet.HiddenStatesBias;
			for (var j = 0; j < _hiddenStatesCount; j++) {
				var lastDerivativeAverageForHiddenBias = _derivativeAveragesForHiddenBias[j];
				var partialDerivativeForHiddenBias = _packageDerivativeForHiddenBias[j];
				_packageDerivativeForHiddenBias[j] = 0f;
				_learnFactorsForHiddenBias[j] = (lastDerivativeAverageForHiddenBias*partialDerivativeForHiddenBias > 0.0f) ?
					Math.Min(_learnFactorsForHiddenBias[j] + _properties.SpeedBonus, _properties.SpeedUpBorder):
					Math.Max(_learnFactorsForHiddenBias[j]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
				_derivativeAveragesForHiddenBias[j] = _properties.AverageLearnFactor*partialDerivativeForHiddenBias + 
					(1f - _properties.AverageLearnFactor)*lastDerivativeAverageForHiddenBias;

				var newDeltaForHiddenBias = curLearnSpeed*_learnFactorsForHiddenBias[j]*partialDerivativeForHiddenBias + 
					                        _properties.Momentum*_oldDeltaWeightsForHiddenBias[j];
				_oldDeltaWeightsForHiddenBias[j] = newDeltaForHiddenBias;
				hiddenStatesBias[j] += (1f + _properties.Momentum)*newDeltaForHiddenBias;
			}

			var labelsBias = _neuralNet.LabelsBias;
			for (var k = 0; k < _labelsCount; k++) {
				var lastDerivativeAverageForLabelBias = _derivativeAveragesForLabelBias[k];
				var partialDerivativeForLabelBias = _packageDerivativeForLabelBias[k];
				_packageDerivativeForLabelBias[k] = 0f;
				_learnFactorsForLabelBias[k] = (lastDerivativeAverageForLabelBias*partialDerivativeForLabelBias > 0.0f) ?
					Math.Min(_learnFactorsForLabelBias[k] + _properties.SpeedBonus, _properties.SpeedUpBorder):
					Math.Max(_learnFactorsForLabelBias[k]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
				_derivativeAveragesForLabelBias[k] = _properties.AverageLearnFactor*partialDerivativeForLabelBias + 
					(1f - _properties.AverageLearnFactor)*lastDerivativeAverageForLabelBias;

				var newDeltaForLabelBias = curLearnSpeed*_learnFactorsForLabelBias[k]*partialDerivativeForLabelBias + 
					                        _properties.Momentum*_oldDeltaWeightsForLabelBias[k];
				_oldDeltaWeightsForLabelBias[k] = newDeltaForLabelBias;
				labelsBias[k] += (1f + _properties.Momentum)*newDeltaForLabelBias;
			}
		}
	}
}
