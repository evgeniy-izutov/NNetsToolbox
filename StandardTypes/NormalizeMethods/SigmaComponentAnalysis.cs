using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.Distributions;

namespace StandardTypes {
	[Serializable]
	public sealed class SigmaComponentAnalysis : INormalizeMethod {
		private readonly IInvertibleFunction _normalizationFunction;
		private float[] _inputMeanValueVector;
		private float[] _inputSigmaVector;
		private float[] _outputMeanValueVector;
		private float[] _outputSigmaVector;
		private int _inputVectorSize;
		private int _outputVectorSize;
		private int _dataSize;
		private bool _isPossibleNormalize;
		private bool _isNormalizeOutput;
		private bool _isSkipGaps;

		public SigmaComponentAnalysis(IInvertibleFunction normalizationFunction = null) {
			_normalizationFunction = normalizationFunction;
			_isPossibleNormalize = false;
			_isNormalizeOutput = false;
			_isSkipGaps = false;
		}

		public void CollectStatistics(IList<TrainSingle> data) {
			PrepareData(data);
			CalcProbabilisticProperties(data);
			_isPossibleNormalize = true;
		}

		public void CollectStatistics(IList<TrainPair> data) {
			PrepareData(data);
			CalcProbabilisticProperties(data);
			_isPossibleNormalize = true;
		}

		public void FillRandomlyGaps(IList<TrainSingle> data) {
			var normalGenerator = new Normal();
			normalGenerator.RandomSource = new Random();
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				if (example.MissedInputIndexes != null) {
					RandomlyFillVector(normalGenerator, example.Input, _inputMeanValueVector, _inputSigmaVector, example.MissedInputIndexes);
				}
			}
		}

		public void FillRandomlyGaps(IList<TrainPair> data) {
			var normalGenerator = new Normal();
			normalGenerator.RandomSource = new Random();
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				if (example.MissedInputIndexes != null) {
					RandomlyFillVector(normalGenerator, example.Input, _inputMeanValueVector, _inputSigmaVector, example.MissedInputIndexes);
				}
			}
		}

		public void NormalizeSet(IList<TrainSingle> data, bool isSkipGaps) {
			_isSkipGaps = isSkipGaps;
			if (_isPossibleNormalize) {
				ChangeValues(data);
			}
		}

		public void NormalizeSet(IList<TrainPair> data, bool isNormalizeOutput, bool isSkipGaps) {
			_isNormalizeOutput = isNormalizeOutput;
			_isSkipGaps = isSkipGaps;
			if (_isPossibleNormalize) {
				ChangeValues(data);
			}
		}

		public float[] NormalizeInputVector(float[] inputVector, HashSet<int> missedInputIndexes) {
			if (!_isPossibleNormalize) {
				throw new Exception("MeanValue and sigma dosn't calculated.");
			}

			var newVector = new float[_inputVectorSize];
			for (var i = 0; i < _inputVectorSize; i++) {
				if (_isSkipGaps && (missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					newVector[i] = 0.0f;
				}
				else {
					var sigma = _inputSigmaVector[i];
					if (Math.Abs(sigma) < float.Epsilon) {
						newVector[i] = 0.0f;
					}
					else {
						if (_normalizationFunction != null) {
							newVector[i] = _normalizationFunction.Calculate((inputVector[i] - _inputMeanValueVector[i])/sigma);
						}
						else {
							newVector[i] = (inputVector[i] - _inputMeanValueVector[i])/sigma;
						}
					}
				}
			}

			return newVector;
		}

		public float[] DenormalizeInputVector(float[] normalizedInputVector, HashSet<int> missedInputIndexes) {
			if (!_isPossibleNormalize) {
				throw new Exception("MeanValue and sigma dosn't calculated.");
			}

			var newVector = new float[_inputVectorSize];
			for (var i = 0; i < _inputVectorSize; i++) {
				if (_isSkipGaps && (missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					newVector[i] = 0.0f;
				}
				else {
					var sigma = _inputSigmaVector[i];
					if (Math.Abs(sigma) < float.Epsilon) {
						newVector[i] = _inputMeanValueVector[i];
					}
					else {
						if (_normalizationFunction != null) {
							newVector[i] = _normalizationFunction.Calculate(sigma*normalizedInputVector[i] + _inputMeanValueVector[i]);
						}
						else {
							newVector[i] = sigma*normalizedInputVector[i] + _inputMeanValueVector[i];
						}
					}
				}
			}

			return newVector;
		}

		public void DenormalizeOutputVector(float[] outputVector) {
			if (!_isPossibleNormalize) {
				throw new Exception("MeanValue and sigma dosn't calculated.");
			}

			if (!_isNormalizeOutput) {
				return;
			}

			for (var i = 0; i < _outputVectorSize; i++) {
				var meanValue = _outputMeanValueVector[i];
				var sigma = _outputSigmaVector[i];
				var value = outputVector[i];
				if (_normalizationFunction != null) {
					outputVector[i] = _normalizationFunction.CalculateInvers(value)*sigma + meanValue;
				}
				else {
					outputVector[i] = value*sigma + meanValue;
				}
			}
		}

		public void Save(string outputPath) {
			if (_isPossibleNormalize) {
				var outputStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
				var serializer = new BinaryFormatter();
				serializer.Serialize(outputStream, _inputVectorSize);
				serializer.Serialize(outputStream, _outputVectorSize);
				serializer.Serialize(outputStream, _inputMeanValueVector);
				serializer.Serialize(outputStream, _inputSigmaVector);
				serializer.Serialize(outputStream, _outputMeanValueVector);
				serializer.Serialize(outputStream, _outputSigmaVector);
				serializer.Serialize(outputStream, _isNormalizeOutput);
				outputStream.Close();
			}
			else {
				throw new Exception("No any data to save");
			}
		}

		public void Load(string inputPath) {
			var inputStream = new FileStream(inputPath, FileMode.Open, FileAccess.Read, FileShare.Read);
			var deserializer = new BinaryFormatter();
			_inputVectorSize = (int) deserializer.Deserialize(inputStream);
			_outputVectorSize = (int) deserializer.Deserialize(inputStream);

			_inputMeanValueVector = (float[]) deserializer.Deserialize(inputStream);
			_inputSigmaVector = (float[]) deserializer.Deserialize(inputStream);
			_outputMeanValueVector = (float[]) deserializer.Deserialize(inputStream);
			_outputSigmaVector = (float[]) deserializer.Deserialize(inputStream);
			_isNormalizeOutput = (bool) deserializer.Deserialize(inputStream);

			inputStream.Close();

			_isPossibleNormalize = true;
		}

		public int InputVectorSize {
			get { return _inputVectorSize; }
		}

		public int OutputVectorSize {
			get { return _outputVectorSize; }
		}

		private void PrepareData(IList<TrainSingle> data) {
			_dataSize = data.Count;
			
			_inputVectorSize = data[0].Input.Length;
			_inputMeanValueVector = new float[_inputVectorSize];
			_inputSigmaVector = new float[_inputVectorSize];

			_outputVectorSize = 0;
		}

		private void PrepareData(IList<TrainPair> data) {
			_dataSize = data.Count;
			
			_inputVectorSize = data[0].Input.Length;
			_inputMeanValueVector = new float[_inputVectorSize];
			_inputSigmaVector = new float[_inputVectorSize];

			_outputVectorSize = data[0].Output.Length;
			_outputMeanValueVector = new float[_outputVectorSize];
			_outputSigmaVector = new float[_outputVectorSize];
		}

		private void CalcProbabilisticProperties(IList<TrainSingle> data) {
			var realDataSizes = new int[_inputVectorSize];
			for (var i = 0; i < _dataSize; i++) {
				var inputVector = data[i].Input;
				var missedInputIndexes = data[i].MissedInputIndexes;
				for (var j = 0; j < _inputVectorSize; j++) {
					if ((missedInputIndexes == null) || (!missedInputIndexes.Contains(j))) {
						var value = inputVector[j];
						_inputMeanValueVector[j] += value;
						_inputSigmaVector[j] += value*value;
						realDataSizes[j]++;
					}
				}
			}

			for (var i = 0; i < _inputVectorSize; i++) {
				var dataSize = realDataSizes[i];
				var meanValueFactorForInput = 1.0f/dataSize;
				var sigmaFactorForInput = dataSize/(dataSize - 1.0f);
				_inputMeanValueVector[i] *= meanValueFactorForInput;
				var meanValue = _inputMeanValueVector[i];
				var sumSqrValues = _inputSigmaVector[i];
				_inputSigmaVector[i] = (float) Math.Sqrt(sigmaFactorForInput*(meanValueFactorForInput*sumSqrValues - meanValue*meanValue));
			}
		}

		private void CalcProbabilisticProperties(IList<TrainPair> data) {
			var realDataSizes = new int[_inputVectorSize];
			for (var i = 0; i < _dataSize; i++) {
				var inputVector = data[i].Input;
				var missedInputIndexes = data[i].MissedInputIndexes;
				for (var j = 0; j < _inputVectorSize; j++) {
					if ((missedInputIndexes == null) || (!missedInputIndexes.Contains(j))) {
						var value = inputVector[j];
						_inputMeanValueVector[j] += value;
						_inputSigmaVector[j] += value*value;
						realDataSizes[j]++;
					}
				}

				var outputVector = data[i].Output;
				for (var j = 0; j < _outputVectorSize; j++) {
					var value = outputVector[j];
					_outputMeanValueVector[j] += value;
					_outputSigmaVector[j] += value*value;
				}
			}

			for (var i = 0; i < _inputVectorSize; i++) {
				var dataSize = realDataSizes[i];
				var meanValueFactorForInput = 1.0f/dataSize;
				var sigmaFactorForInput = dataSize/(dataSize - 1.0f);
				_inputMeanValueVector[i] *= meanValueFactorForInput;
				var meanValue = _inputMeanValueVector[i];
				var sumSqrValues = _inputSigmaVector[i];
				_inputSigmaVector[i] = (float) Math.Sqrt(sigmaFactorForInput*(meanValueFactorForInput*sumSqrValues - meanValue*meanValue));
			}
			var meanValueFactor = 1.0f/_dataSize;
			var sigmaFactor = _dataSize/(_dataSize - 1.0f);
			for (var i = 0; i < _outputVectorSize; i++) {
				_outputMeanValueVector[i] *= meanValueFactor;
				var meanValue = _outputMeanValueVector[i];
				var sumSqrValues = _outputSigmaVector[i];
				_outputSigmaVector[i] = (float) Math.Sqrt(sigmaFactor*(meanValueFactor*sumSqrValues - meanValue*meanValue));
			}
		}

		private void ChangeValues(IList<TrainSingle> data) {
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				DirectConversationVector(example.Input, _inputMeanValueVector, _inputSigmaVector, example.MissedInputIndexes);
			}
		}

		private void ChangeValues(IList<TrainPair> data) {
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				DirectConversationVector(example.Input, _inputMeanValueVector, _inputSigmaVector, example.MissedInputIndexes);
				if (_isNormalizeOutput) {
					DirectConversationVector(example.Output, _outputMeanValueVector, _outputSigmaVector);
				}
			}
		}

		private void DirectConversationVector(float[] vector, float[] meanValueVector, float[] sigmaValueVector, HashSet<int> missedInputIndexes = null) {
			var length = vector.Length;
			for (var i = 0; i < length; i++) {
				if (_isSkipGaps && (missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					vector[i] = 0.0f;
				}
				else {
					var sigma = sigmaValueVector[i];
					if (Math.Abs(sigma) < float.Epsilon) {
						vector[i] = 0.0f;
					}
					else {
						var value = vector[i];
						if (_normalizationFunction != null) {
							vector[i] = _normalizationFunction.Calculate((value - meanValueVector[i])/sigma);
						}
						else {
							vector[i] = (value - meanValueVector[i])/sigma;
						}
					}
				}
			}
		}

		private static void RandomlyFillVector(Normal random, float[] vector, float[] meanValueVector, float[] sigmaValueVector, HashSet<int> missedInputIndexes) {
			foreach (var i in missedInputIndexes) {
				vector[i] = meanValueVector[i] + ((float) random.Sample())*sigmaValueVector[i];
			}
		}
	}
}