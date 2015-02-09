using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace StandardTypes {
	[Serializable]
	public sealed class MinMaxComponentAnalysis : INormalizeMethod {
		private readonly IInvertibleFunction _normalizationFunction;
		private float[] _inputMinVector;
		private float[] _inputMaxVector;
		private float[] _outputMinVector;
		private float[] _outputMaxVector;
		private int _inputVectorSize;
		private int _outputVectorSize;
		private int _dataSize;
		private bool _isPossibleNormalize;
		private bool _isNormalizeOutput;
		private bool _isSkipGaps;

		public MinMaxComponentAnalysis(IInvertibleFunction normalizationFunction = null) {
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
			var uniformGenerator = new Random();
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				if (example.MissedInputIndexes != null) {
					RandomlyFillVector(uniformGenerator, example.Input, _inputMinVector, _inputMaxVector, example.MissedInputIndexes);
				}
			}
		}

		public void FillRandomlyGaps(IList<TrainPair> data) {
			var uniformGenerator = new Random();
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				if (example.MissedInputIndexes != null) {
					RandomlyFillVector(uniformGenerator, example.Input, _inputMinVector, _inputMaxVector, example.MissedInputIndexes);
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
				throw new Exception("Min and max values dosn't calculated.");
			}

			var newVector = new float[_inputVectorSize];
			for (var i = 0; i < _inputVectorSize; i++) {
				if ((_isSkipGaps) && (missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					newVector[i] = 0.0f;
				}
				else {
					var minValue = _inputMinVector[i];
					var maxValue = _inputMaxVector[i];
					var dif = maxValue - minValue;
					if (Math.Abs(dif) < float.Epsilon) {
						newVector[i] = minValue;
					}
					else {
						if (_normalizationFunction != null) {
							newVector[i] = _normalizationFunction.Calculate((inputVector[i] - minValue)/dif);
						}
						else {
							newVector[i] = (inputVector[i] - minValue)/dif;
						}
					}
				}
			}

			return newVector;
		}

		public float[] DenormalizeInputVector(float[] normalizedInputVector, HashSet<int> missedInputIndexes = null) {
			if (!_isPossibleNormalize) {
				throw new Exception("MeanValue and sigma dosn't calculated.");
			}

			var newVector = new float[_inputVectorSize];
			for (var i = 0; i < _inputVectorSize; i++) {
				if ((missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					newVector[i] = 0.0f;
				}
				else {
					var minValue = _inputMinVector[i];
					var maxValue = _inputMaxVector[i];
					var dif = maxValue - minValue;
					if (Math.Abs(dif) < float.Epsilon) {
						newVector[i] = minValue;
					}
					else {
						if (_normalizationFunction != null) {
							newVector[i] = _normalizationFunction.Calculate(dif*normalizedInputVector[i] + minValue);
						}
						else {
							newVector[i] = dif*normalizedInputVector[i] + minValue;
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
				var minValue = _outputMinVector[i];
				var maxValue = _outputMaxVector[i];
				var dif = maxValue - minValue;
				var value = outputVector[i];
				if (_normalizationFunction != null) {
					outputVector[i] = _normalizationFunction.CalculateInvers(value)*dif + minValue;
				}
				else {
					outputVector[i] = value*dif + minValue;
				}
			}
		}

		public void Save(string outputPath) {
			if (_isPossibleNormalize) {
				var outputStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
				var serializer = new BinaryFormatter();
				serializer.Serialize(outputStream, _inputVectorSize);
				serializer.Serialize(outputStream, _outputVectorSize);
				serializer.Serialize(outputStream, _inputMinVector);
				serializer.Serialize(outputStream, _inputMaxVector);
				serializer.Serialize(outputStream, _outputMinVector);
				serializer.Serialize(outputStream, _outputMaxVector);
				serializer.Serialize(outputStream, _isNormalizeOutput);
				outputStream.Close();
			}
			else {
				throw new Exception("No any data to save");
			}
		}

		public void Load(string inputPath) {
			var inputStream = new FileStream(inputPath, FileMode.Open,
			                                 FileAccess.Read, FileShare.Read);
			var deserializer = new BinaryFormatter();
			_inputVectorSize = (int) deserializer.Deserialize(inputStream);
			_outputVectorSize = (int) deserializer.Deserialize(inputStream);

			_inputMinVector = (float[]) deserializer.Deserialize(inputStream);
			_inputMaxVector = (float[]) deserializer.Deserialize(inputStream);
			_outputMinVector = (float[]) deserializer.Deserialize(inputStream);
			_outputMaxVector = (float[]) deserializer.Deserialize(inputStream);
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
			_inputMinVector = new float[_inputVectorSize];
			_inputMaxVector = new float[_inputVectorSize];

			_outputVectorSize = 0;
		}
		
		private void PrepareData(IList<TrainPair> data) {
			_dataSize = data.Count;
			
			_inputVectorSize = data[0].Input.Length;
			_inputMinVector = new float[_inputVectorSize];
			_inputMaxVector = new float[_inputVectorSize];

			_outputVectorSize = data[0].Output.Length;
			_outputMinVector = new float[_outputVectorSize];
			_outputMaxVector = new float[_outputVectorSize];
		}

		private void CalcProbabilisticProperties(IList<TrainSingle> data) {
			for (var i = 0; i < _dataSize; i++) {
				var inputVector = data[i].Input;
				var missedInputIndexes = data[i].MissedInputIndexes;
				for (var j = 0; j < _inputVectorSize; j++) {
					if ((missedInputIndexes == null) || (!missedInputIndexes.Contains(j))) {
						var value = inputVector[j];
						if (value < _inputMinVector[j]) {
							_inputMinVector[j] = value;
						}
						else if (value > _inputMaxVector[j]) {
							_inputMaxVector[j] = value;
						}
					}
				}
			}
		}

		private void CalcProbabilisticProperties(IList<TrainPair> data) {
			for (var i = 0; i < _dataSize; i++) {
				var inputVector = data[i].Input;
				var missedInputIndexes = data[i].MissedInputIndexes;
				for (var j = 0; j < _inputVectorSize; j++) {
					if ((missedInputIndexes == null) || (!missedInputIndexes.Contains(j))) {
						var value = inputVector[j];
						if (value < _inputMinVector[j]) {
							_inputMinVector[j] = value;
						}
						else if (value > _inputMaxVector[j]) {
							_inputMaxVector[j] = value;
						}
					}
				}

				var outputVector = data[i].Output;
				for (var j = 0; j < _outputVectorSize; j++) {
					var value = outputVector[j];
					if (value < _outputMinVector[j]) {
						_outputMinVector[j] = value;
					}
					else if (value > _outputMaxVector[j]) {
						_outputMaxVector[j] = value;
					}
				}
			}
		}

		private void ChangeValues(IList<TrainSingle> data) {
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				DirectConversationVector(example.Input, _inputMinVector, _inputMaxVector, example.MissedInputIndexes);
			}
		}

		private void ChangeValues(IList<TrainPair> data) {
			for (var i = 0; i < data.Count; i++) {
				var example = data[i];
				DirectConversationVector(example.Input, _inputMinVector, _inputMaxVector, example.MissedInputIndexes);
				if (_isNormalizeOutput) {
					DirectConversationVector(example.Output, _outputMinVector, _outputMaxVector);
				}
			}
		}

		private void DirectConversationVector(float[] vector, float[] minVector, float[] maxVector, HashSet<int> missedInputIndexes = null) {
			for (var i = 0; i < vector.Length; i++) {
				if ((_isSkipGaps) && (missedInputIndexes != null) && (missedInputIndexes.Contains(i))) {
					vector[i] = 0.0f;
				}
				else {
					var minValue = minVector[i];
					var maxValue = maxVector[i];
					var dif = maxValue - minValue;
					if (Math.Abs(dif) < float.Epsilon) {
						vector[i] = 0.0f;
					}
					else {
						var value = vector[i];
						if (_normalizationFunction != null) {
							vector[i] = _normalizationFunction.Calculate((value - minValue)/dif);
						}
						else {
							vector[i] = (value - minValue)/dif;
						}
					}
				}
			}
		}

		private static void RandomlyFillVector(Random random, float[] vector, float[] minVector, float[] maxVector, HashSet<int> missedInputIndexes) {
			foreach (var i in missedInputIndexes) {
				vector[i] = minVector[i] + (maxVector[i] - minVector[i])*((float) random.NextDouble());
			}
		}
	}
}