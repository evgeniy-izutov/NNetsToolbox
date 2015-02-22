using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using StandardTypes;

namespace NeuralNet.MultyLayerPerceptron {
	public sealed class BackPropagationAlgorithm : TrainMethod {
		private readonly RandomAccessIterator<TrainPair> _trainDataIterator;
        private readonly IList<TrainPair> _testData;
		private ITrainProperties _properties;
		private MultyLayerPerceptron _neuralNet;
		private BaseNeuralBlock[] _layers;
		private int[] _neuronNetLayersStruct;
		private float _packageFactor;
		private int _outputLayerNum;
		private float[] _neuronNetOutput;
		private float[] _neuronNetInput;
		private float[][] _oldDeltaWeights;
		private float[][] _derivativeAverages;
		private float[][] _packageDerivative;
		private float[][] _learnFactors;
		private float[][] _oldDeltaWeightsForBias;
		private float[][] _derivativeAveragesForBias;
		private float[][] _packageDerivativeForBias;
		private float[][] _learnFactorsForBias;
		private float[] _gradients;
		private float[] _gradientsIntermediate;
		private int _epochNumber;
		private int _packagesCount;

		public BackPropagationAlgorithm(IList<TrainPair> trainData) {
			_trainDataIterator = new RandomAccessIterator<TrainPair>(trainData);
		}

        public BackPropagationAlgorithm(IList<TrainPair> trainData, IList<TrainPair> testData) {
            _trainDataIterator = new RandomAccessIterator<TrainPair>(trainData);
            _testData = testData;
        }

		public override void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties) {
			if (!(neuralNet is MultyLayerPerceptron)) {
				throw new ArgumentException("Neural net has other structure");
			}
			_neuralNet = (MultyLayerPerceptron) neuralNet;
			_properties = trainProperties;
			_layers = _neuralNet.Layers;
			_neuronNetLayersStruct = _neuralNet.GetLayersStruct();
			_outputLayerNum = _layers.Length - 1;
			_packageFactor = 1.0f/_properties.PackageSize;
			_packagesCount = CalculatePackagesCount();
			AllocateMemory();
			ProcessSate = IterativeProcessState.NotStarted;
		}

		public override ITrainProperties Properties {
			get { return _properties; }
		}

		private void AllocateMemory() {
			int maxGradientSize;
			FindMaxSize(out maxGradientSize);

			_gradients = new float[maxGradientSize];
			_gradientsIntermediate = new float[maxGradientSize];

			_neuronNetOutput = new float[_layers[_outputLayerNum].Size];

			_oldDeltaWeights = new float[_layers.Length][];
			_derivativeAverages = new float[_layers.Length][];
			_packageDerivative = new float[_layers.Length][];
			_learnFactors = new float[_layers.Length][];
			for (var i = 0; i < _layers.Length; i++) {
				var weightsCount = _layers[i].GetWeights()[0].Length;
				_oldDeltaWeights[i] = new float[weightsCount];
				_derivativeAverages[i] = new float[weightsCount];
				_packageDerivative[i] = new float[weightsCount];
				_learnFactors[i] = new float[weightsCount];
				for (var j = 0; j < weightsCount; j++) {
					_learnFactors[i][j] = 1f;
				}
			}

			_oldDeltaWeightsForBias = new float[_layers.Length][];
			_derivativeAveragesForBias = new float[_layers.Length][];
			_packageDerivativeForBias = new float[_layers.Length][];
			_learnFactorsForBias = new float[_layers.Length][];
			for (var i = 0; i < _layers.Length; i++) {
				var layerSize = _layers[i].Size;
				_oldDeltaWeightsForBias[i] = new float[layerSize];
				_derivativeAveragesForBias[i] = new float[layerSize];
				_packageDerivativeForBias[i] = new float[layerSize];
				_learnFactorsForBias[i] = new float[layerSize];
				for (var j = 0; j < layerSize; j++) {
					_learnFactorsForBias[i][j] = 1f;
				}
			}
		}

		private void FindMaxSize(out int maxGradientSize) {
			maxGradientSize = 0;
			for (var i = 1; i < _neuronNetLayersStruct.Length; i++) {
				var curLayerSize = _neuronNetLayersStruct[i];
				if (curLayerSize > maxGradientSize) {
					maxGradientSize = curLayerSize;
				}
			}
		}

		protected override void RunIterativeProcess() {
			var trainErrorValue = _properties.Epsilon + 1.0f;
			var testErrorValue = float.NaN;
			var minTestErrorValue = float.MaxValue;
			_epochNumber = 1;
			while ((ProcessSate == IterativeProcessState.InProgress) && 
				(trainErrorValue > _properties.Epsilon) && 
				(_epochNumber <= _properties.MaxIterationCount) && 
				(float.IsNaN(testErrorValue) || Math.Abs(testErrorValue - minTestErrorValue) < _properties.CvLimit)) {
				
				TrainEpoch();

				trainErrorValue = TestModel(_trainDataIterator.Collection);
			    testErrorValue = TestModel(_testData);
				if (!float.IsNaN(testErrorValue) && (testErrorValue < minTestErrorValue)) {
					minTestErrorValue = testErrorValue;
				}

                OnIterationCompleted(new IterationCompletedEventArgs(_epochNumber, trainErrorValue, testErrorValue));
				_epochNumber++;
			}
			OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(_epochNumber));
		}

		protected override void ApplyResults() {
			if (ProcessSate == IterativeProcessState.Finished) {
				ClearReference();
			}
		}

		private void ClearReference() {
			_properties = null;
			_neuralNet = null;
			_layers = null;
			_neuronNetLayersStruct = null;
			_neuronNetOutput = null;
			_neuronNetInput = null;
			_oldDeltaWeights = null;
			_gradients = null;
			_gradientsIntermediate = null;
			_derivativeAverages = null;
			_packageDerivative = null;
			_oldDeltaWeightsForBias = null;
			_derivativeAveragesForBias = null;
			_packageDerivativeForBias = null;
			_learnFactors = null;
			_learnFactorsForBias = null;
		}

		private int CalculatePackagesCount() {
			var packagesCount = _trainDataIterator.Size()/_properties.PackageSize;
			if (_trainDataIterator.Size()%_properties.PackageSize != 0) {
				packagesCount++;
			}
			return packagesCount;
		}

	    private float TestModel(IList<TrainPair> data) {
            if (data == null || data.Count == 0) {
	            return float.NaN;
	        }
	        else {
                var sumError = 0.0f;
                for (var i = 0; i < data.Count; i++) {
                    var testPair = data[i];
                    _neuronNetInput = testPair.Input;
                    _neuralNet.Predict(_neuronNetInput, _neuronNetOutput);
                    sumError += _properties.Metrics.Calculate(testPair.Output, _neuronNetOutput);
                }
                return sumError / data.Count;
	        }
	    }

		private void TrainEpoch() {
			_trainDataIterator.RefreshRandomAccess();
			for (var i = 0; i < _packagesCount; i++) {
				TrainPackage();
			}
		}

		private void TrainPackage() {
			for (var i = 0; i < _properties.PackageSize; i++) {
				var trainPair = _trainDataIterator.Next();
				_neuronNetInput = trainPair.Input;
				_neuralNet.Predict(_neuronNetInput, _neuronNetOutput);
				var outputPartialDerivaitves = _properties.Metrics.CalculatePartialDerivaitve(trainPair.Output, _neuronNetOutput);
				CollectWeightsDelta(outputPartialDerivaitves);
			}
			ModifyWeightsOfNeuronNet();
		}

		private void CollectWeightsDelta(float[] outputPartialDerivaitves) {
			const int firstLayerNumber = 0;
			var lastHiddenLayerNumber = _outputLayerNum - 1;

			CollectWeightsDeltaOfLayer(lastHiddenLayerNumber + 1, LocalGradientForOutputLayer, outputPartialDerivaitves);
			for (var layerNumber = lastHiddenLayerNumber; layerNumber >= firstLayerNumber; layerNumber--) {
				CollectWeightsDeltaOfLayer(layerNumber, LocalGradientForHiddenLayer, null);
			}
		}

		private void CollectWeightsDeltaOfLayer(int layerNum, LocalGradientDelegate localGradientfunction, float[] outputPartialDerivaitves) {
			var curLayer = _layers[layerNum];
			var prevLayerState = (layerNum > 0) ? _layers[layerNum - 1].GetState() : _neuronNetInput;
			var curGradients = _gradientsIntermediate;
			var nextGradients = _gradients;
			var nextLayerSize = (layerNum < _outputLayerNum) ? _neuronNetLayersStruct[layerNum + 2] : 0;
			var nextLayerWeights = (layerNum < _outputLayerNum) ? _layers[layerNum + 1].GetWeights()[0] : null;
			var packageDerivative = _packageDerivative[layerNum];
			var packageDerivativeForBias = _packageDerivativeForBias[layerNum];

			localGradientfunction(curGradients, curLayer.GetActivationFunction(), curLayer.GetState(), 
				outputPartialDerivaitves, nextGradients, nextLayerWeights, nextLayerSize);
			Parallel.For(0, curLayer.Size, i => {
				var localGradient = curGradients[i];
				for (var j = 0; j < prevLayerState.Length; j++) {
					packageDerivative[prevLayerState.Length*i + j] -= localGradient*prevLayerState[j];
				}
				packageDerivativeForBias[i] -= localGradient;
			});
			_gradients = curGradients;
			_gradientsIntermediate = nextGradients;
		}

		private void ModifyWeightsOfNeuronNet() {
			const int firstLayerNumber = 0;
			for (var layerNum = firstLayerNumber; layerNum <= _outputLayerNum; layerNum++) {
				var curLayer = _layers[layerNum];
				var curLayerWeights = curLayer.GetWeights()[0];
				var curLayerBias = curLayer.GetBias();
				var learnFactors = _learnFactors[layerNum];
				var packageDerivative = _packageDerivative[layerNum];
				var derivativeAverages = _derivativeAverages[layerNum];
				var oldDeltaWeights = _oldDeltaWeights[layerNum];
				var learnFactorsForBias = _learnFactorsForBias[layerNum];
				var packageDerivativeForBias = _packageDerivativeForBias[layerNum];
				var oldDeltaWeightsForBias = _oldDeltaWeightsForBias[layerNum];
				var derivativeAveragesForBias = _derivativeAveragesForBias[layerNum];
				var prevLayerSize = _neuronNetLayersStruct[layerNum];
				var curLearnSpeed = _properties.BaseLearnSpeed*_properties.LearnFactorStrategy.GetFactor(_epochNumber);

				Parallel.For(0, curLayer.Size, i => {
					for (var j = 0; j < prevLayerSize; j++) {
						var weightIndex = prevLayerSize*i + j;

						var lastDerivativeAverage = derivativeAverages[weightIndex];
						var partialDerivative = _packageFactor*packageDerivative[weightIndex] - 
							_properties.Regularization.GetDerivative(curLayerWeights[weightIndex]);
						packageDerivative[weightIndex] = 0.0f;
						learnFactors[weightIndex] = lastDerivativeAverage*partialDerivative > 0.0f ?
							Math.Min(learnFactors[weightIndex] + _properties.SpeedBonus, _properties.SpeedUpBorder) :
							Math.Max(learnFactors[weightIndex]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
						derivativeAverages[weightIndex] = _properties.AverageLearnFactor*partialDerivative + 
							(1.0f - _properties.AverageLearnFactor)*lastDerivativeAverage;

						var oldDelta = _properties.Momentum*oldDeltaWeights[weightIndex];
						var newDelta = curLearnSpeed*learnFactors[weightIndex]*partialDerivative + oldDelta;
						oldDeltaWeights[weightIndex] = newDelta;
						curLayerWeights[weightIndex] += (1f + _properties.Momentum)*newDelta;
					}

					var lastDerivativeAverageForBias = derivativeAveragesForBias[i];
					var partialDerivativeForBias = _packageFactor*packageDerivativeForBias[i];
					packageDerivativeForBias[i] = 0.0f;
					learnFactorsForBias[i] = lastDerivativeAverageForBias*partialDerivativeForBias > 0.0f ? 
						Math.Min(learnFactorsForBias[i] + _properties.SpeedBonus, _properties.SpeedUpBorder) : 
						Math.Max(learnFactorsForBias[i]*_properties.SpeedPenalty, _properties.SpeedLowBorder);
					derivativeAveragesForBias[i] = _properties.AverageLearnFactor*partialDerivativeForBias + 
						(1.0f - _properties.AverageLearnFactor)*lastDerivativeAverageForBias;

					var oldDeltaForBias = _properties.Momentum*oldDeltaWeightsForBias[i];
					var newDeltaForBias = curLearnSpeed*partialDerivativeForBias + oldDeltaForBias;
					oldDeltaWeightsForBias[i] = newDeltaForBias;
					curLayerBias[i] += (1f + _properties.Momentum)*newDeltaForBias;
				});
			}
		}

		private static void LocalGradientForOutputLayer(float[] gradientsOutput, IActivationFunction function, float[] state, float[] errors,
			float[] nextLayerGradients, float[] nextLayerOldWeights, int nextLayerSize) {

			function.CalculateFirstDerivative(gradientsOutput, errors, state);
		}

		private static void LocalGradientForHiddenLayer(float[] gradientsOutput, IActivationFunction function, float[] state, float[] errors,
			float[] nextLayerGradients, float[] nextLayerOldWeights, int nextLayerSize) {
			var curLayerSize = state.Length;
			
			for (var neuronNum = 0; neuronNum < curLayerSize; neuronNum++) {
				gradientsOutput[neuronNum] = 0.0f;
			}

			for (var j = 0; j < nextLayerSize; j++) {
				var nextLayerGradient = nextLayerGradients[j];
				for (var neuronNum = 0; neuronNum < curLayerSize; neuronNum++) {
					gradientsOutput[neuronNum] += nextLayerGradient*nextLayerOldWeights[curLayerSize*j + neuronNum];
				}
			}

			function.CalculateFirstDerivative(gradientsOutput, state);
		}

		private delegate void LocalGradientDelegate(float[] gradientsOutput, IActivationFunction function, float[] state, float[] errors,
			float[] nextLayerGradients, float[] nextLayerOldWeights, int nextLayerSize);
	}
}