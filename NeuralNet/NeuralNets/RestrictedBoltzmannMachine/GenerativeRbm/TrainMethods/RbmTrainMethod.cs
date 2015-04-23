using System;
using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.GenerativeRbm {
	public abstract class RbmTrainMethod : TrainMethod<TrainSingle> {
		private readonly RandomAccessIterator<TrainSingle> _trainDataIterator;
		private readonly IList<TrainSingle> _testData;
		private readonly IGradientFunction _gradientFunction;
		private float[] _neuronNetOutput;
        private float packageFactor;
		protected RbmGradients gradients;
		protected ITrainProperties<TrainSingle> properties;
		protected RestrictedBoltzmannMachine neuralNet;
		protected int visibleStatesCount;
		protected int hiddenStatesCount;
		protected int epochNumber;
		protected int packagesCount;

		protected RbmTrainMethod(IList<TrainSingle> trainData, IGradientFunction gradient) {
			_gradientFunction = gradient;
			_trainDataIterator = new RandomAccessIterator<TrainSingle>(trainData);
		}

		protected RbmTrainMethod(IList<TrainSingle> trainData, IList<TrainSingle> testData, IGradientFunction gradient) {
			_trainDataIterator = new RandomAccessIterator<TrainSingle>(trainData);
			_testData = testData;
			_gradientFunction = gradient;
		}

		public override void InitilazeMethod(INeuralNet neuralNet, ITrainProperties<TrainSingle> trainProperties) {
			if (!(neuralNet is RestrictedBoltzmannMachine)) {
				throw new ArgumentException("Neural net has other structure");
			}

			this.neuralNet = (RestrictedBoltzmannMachine) neuralNet;
			visibleStatesCount = this.neuralNet.VisibleStates.Length;
			hiddenStatesCount = this.neuralNet.HiddenStates.Length;

			gradients = new RbmGradients(visibleStatesCount, hiddenStatesCount);
			_gradientFunction.Initialize(gradients);

			_neuronNetOutput = new float[visibleStatesCount];

			properties = trainProperties;
			packageFactor = 1.0f/properties.PackageSize;
			packagesCount = CalculatePackagesCount();

			AllocateMemory();

			ProcessSate = IterativeProcessState.NotStarted;
		}

		protected override void RunIterativeProcess() {
			if (IsTestDataAvailable()) {
	            RunTraingWithTesting();
	        }
			else {
				RunTraingWithoutTesting();
			}
		}

		protected override void ApplyResults() {
			if (ProcessSate == IterativeProcessState.Finished) {
				ClearReference();
			}
		}

		public override ITrainProperties<TrainSingle> Properties {
			get { return properties; }
		}

		protected abstract void AllocateMemory();

		protected abstract void ClearReference();

		private bool IsTestDataAvailable() {
			return !(_testData == null || _testData.Count == 0);
		}

		private void RunTraingWithTesting() {
			var trainError = TestModel(_trainDataIterator.Collection);
			var slidingTestError = TestModel(_testData);
			var minTestError = slidingTestError;
			epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) &&
				   (trainError > properties.Epsilon) &&
				   (epochNumber <= properties.MaxIterationCount) &&
				   ((epochNumber <= properties.SkipCvLimitFirstIterations) ||
				    (Math.Abs(slidingTestError - minTestError) < properties.CvLimit))) {
				
				TrainEpoch();

				trainError = TestModel(_trainDataIterator.Collection);
			    var testError = TestModel(_testData);
				slidingTestError = properties.CvSlidingFactor*testError +
					(1f - properties.CvSlidingFactor)*slidingTestError;
				
				if (testError < minTestError) {
					minTestError = testError;
				}

                OnIterationCompleted(new IterationCompletedEventArgs(epochNumber, trainError, testError));
				epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(epochNumber));
		}

		private void RunTraingWithoutTesting() {
			var trainError = TestModel(_trainDataIterator.Collection);
			epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) && 
				   (trainError > properties.Epsilon) && 
				   (epochNumber <= properties.MaxIterationCount)) {
				
				TrainEpoch();

				trainError = TestModel(_trainDataIterator.Collection);

                OnIterationCompleted(new IterationCompletedEventArgs(epochNumber, trainError, float.NaN));
				epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(epochNumber));
		}

		private int CalculatePackagesCount() {
			var count = _trainDataIterator.Size()/properties.PackageSize;
			if (_trainDataIterator.Size()%properties.PackageSize != 0) {
				count++;
			}
			return count;
		}

		private float TestModel(IList<TrainSingle> data) {
			var sumError = 0.0f;
			for (var i = 0; i < data.Count; i++) {
				var testExample = data[i];
				neuralNet.Predict(testExample.Input, _neuronNetOutput);
				sumError += properties.Metrics.Calculate(testExample.Input, _neuronNetOutput);
			}
			return sumError / data.Count;
		}

		private void TrainEpoch() {
			_trainDataIterator.RefreshRandomAccess();
			for (var i = 0; i < packagesCount; i++) {
				TrainPackage(i);
			}
		}

		private void TrainPackage(int packageId) {
			_gradientFunction.PrepareToNextPackage(properties.PackageSize);
			for (var i = 0; i < properties.PackageSize; i++) {
				var input = _trainDataIterator.Next().Input;

				MakePositivePhase(input);
				_gradientFunction.StorePositivePhaseData(input, neuralNet.HiddenStates);
				MakeNegativePhase(packageId);
				_gradientFunction.StoreNegativePhaseData(GetVisibleStatesOnNegativePhase(packageId), GetHiddenStatesOnNegativePhase());
				RestoreVisibleStates(packageId);
			}
			_gradientFunction.MakeGradient(packageFactor);
			ModifyWeightsOfNeuronNet();
		}

		protected abstract void MakePositivePhase(float[] input);

		protected abstract void MakeNegativePhase(int packageId);

		protected abstract float[] GetVisibleStatesOnNegativePhase(int packageId);

		protected abstract float[] GetHiddenStatesOnNegativePhase();

		protected abstract void RestoreVisibleStates(int packageId);

		protected abstract void ModifyWeightsOfNeuronNet();
	}
}
