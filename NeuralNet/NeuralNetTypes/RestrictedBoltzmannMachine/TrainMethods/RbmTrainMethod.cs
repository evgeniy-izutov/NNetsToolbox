using System;
using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public abstract class RbmTrainMethod : TrainMethod {
		private readonly RandomAccessIterator<TrainSingle> _trainDataIterator;
		private readonly IList<TrainSingle> _testData;
		private readonly IGradientFunction _gradientFunction;
		private float[] _neuronNetOutput;
		protected RbmGradients gradients;
		protected ITrainProperties properties;
		protected RestrictedBoltzmannMachine neuralNet;
		protected float packageFactor;
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

		public override void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties) {
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
			var trainErrorValue = properties.Epsilon + 1.0f;
			var testErrorValue = float.NaN;
			var minTestErrorValue = float.MaxValue;
			epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) && 
				(trainErrorValue > properties.Epsilon) && 
				(epochNumber <= properties.MaxIterationCount) && 
				(float.IsNaN(testErrorValue) || Math.Abs(testErrorValue - minTestErrorValue) < properties.CvLimit)) {
				
				TrainEpoch();

				trainErrorValue =  TestModel(_trainDataIterator.Collection);
			    testErrorValue = TestModel(_testData);
				if (!float.IsNaN(testErrorValue) && (testErrorValue < minTestErrorValue)) {
					minTestErrorValue = testErrorValue;
				}

                OnIterationCompleted(new IterationCompletedEventArgs(epochNumber, trainErrorValue, testErrorValue));
				epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(epochNumber));
		}

		protected override void ApplyResults() {
			if (ProcessSate == IterativeProcessState.Finished) {
				ClearReference();
			}
		}

		public override ITrainProperties Properties {
			get { return properties; }
		}

		protected abstract void AllocateMemory();

		protected abstract void ClearReference();

		private int CalculatePackagesCount() {
			var packagesCount = _trainDataIterator.Size()/properties.PackageSize;
			if (_trainDataIterator.Size()%properties.PackageSize != 0) {
				packagesCount++;
			}
			return packagesCount;
		}

		private float TestModel(IList<TrainSingle> data) {
            if (data == null || data.Count == 0) {
	            return float.NaN;
	        }

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