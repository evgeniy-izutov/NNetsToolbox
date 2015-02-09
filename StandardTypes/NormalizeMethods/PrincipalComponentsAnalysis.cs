using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra.Single;

namespace StandardTypes {
	[Serializable]
	public sealed class PrincipalComponentsAnalysis : INormalizeMethod {
		private float[] _inputMeanValueVector;
		private float[] _outputMeanValueVector;
		private float[] _transformMatrix;
		private float[] _singularValues;
		private int _inputVectorSize;
		private int _outputVectorSize;
		private bool _isPossibleNormalize;
		private float _normalizeFactor;
		private float _thresholdRate;
		private int _normalizedInputVectorSize;
		private bool _isNormalizeOutput;
		private bool _isSkipGaps;

		public PrincipalComponentsAnalysis(float thresholdRate) {
			_isPossibleNormalize = false;
			_isNormalizeOutput = false;
			_isSkipGaps = false;
			_thresholdRate = thresholdRate;
		}

		public void CollectStatistics(IList<TrainSingle> data) {
			PrepareData(data);
			CalcMeanValues(data, data.Count);
			SubtractMeanValues(data, data.Count);
			CreateFilter(data);
			_isPossibleNormalize = true;
		}

		public void CollectStatistics(IList<TrainPair> data) {
			PrepareData(data);
			CalcMeanValues(data, data.Count);
			SubtractMeanValues(data, data.Count);
			CreateFilter(data);
			_isPossibleNormalize = true;
		}

		public void FillRandomlyGaps(IList<TrainSingle> data) {
			throw new NotImplementedException();
		}

		public void FillRandomlyGaps(IList<TrainPair> data) {
			throw new NotImplementedException();
		}

		public void NormalizeSet(IList<TrainSingle> data, bool isSkipGaps) {
			_isSkipGaps = isSkipGaps;
			if (_isPossibleNormalize) {
				ApplyFilterForInputComponents(data);
			}
		}

		public void NormalizeSet(IList<TrainPair> data, bool isNormalizeOutput, bool isSkipGaps) {
			_isNormalizeOutput = isNormalizeOutput;
			_isSkipGaps = isSkipGaps;
			if (_isPossibleNormalize) {
				ApplyFilterForInputComponents(data);
			}
		}

		public float[] NormalizeInputVector(float[] inputVector, HashSet<int> missedInputIndexes = null) {
			if (!_isPossibleNormalize) {
				throw new Exception("Mean value dosn't calculated.");
			}

			var newVector = new float[_normalizedInputVectorSize];
			for (var i = 0; i < _normalizedInputVectorSize; i++) {
				var sum = 0.0f;
				for (var k = 0; k < _inputVectorSize; k++) {
					sum += (inputVector[k] - _inputMeanValueVector[k])*_transformMatrix[i*_inputVectorSize + k];
				}
				newVector[i] = sum*_normalizeFactor/_singularValues[i];
			}

			return newVector;
		}

		public float[] DenormalizeInputVector(float[] normalizedInputVector, HashSet<int> missedInputIndexes) {
			if (!_isPossibleNormalize) {
				throw new Exception("Mean value dosn't calculated.");
			}

			var newVector = new float[_normalizedInputVectorSize];

			//for (var i = 0; i < _normalizedInputVectorSize; i++) {
			//	var sum = 0.0f;
			//	for (var k = 0; k < _inputVectorSize; k++) {
			//		sum += (normalizedInputVector[k] - _inputMeanValueVector[k])*_transformMatrix[i*_inputVectorSize + k];
			//	}
			//	newVector[i] = sum*_normalizeFactor/_singularValues[i];
			//}

			return newVector;
		}

		public void DenormalizeOutputVector(float[] outputVector) {
			if (!_isPossibleNormalize) {
				throw new Exception("Mean value dosn't calculated.");
			}

			if (!_isNormalizeOutput) {
				return;
			}

			for (var i = 0; i < _outputVectorSize; i++) {
				outputVector[i] += _outputMeanValueVector[i];
			}
		}

		public void Save(string outputPath) {
			if (_isPossibleNormalize) {
				var outputStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
				var serializer = new BinaryFormatter();
				serializer.Serialize(outputStream, _inputVectorSize);
				serializer.Serialize(outputStream, _outputVectorSize);
				serializer.Serialize(outputStream, _normalizedInputVectorSize);
				serializer.Serialize(outputStream, _inputMeanValueVector);
				serializer.Serialize(outputStream, _outputMeanValueVector);
				serializer.Serialize(outputStream, _transformMatrix);
				serializer.Serialize(outputStream, _singularValues);
				serializer.Serialize(outputStream, _normalizeFactor);
				serializer.Serialize(outputStream, _thresholdRate);
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
			_normalizedInputVectorSize = (int) deserializer.Deserialize(inputStream);

			_inputMeanValueVector = (float[]) deserializer.Deserialize(inputStream);
			_outputMeanValueVector = (float[]) deserializer.Deserialize(inputStream);
			_transformMatrix = (float[]) deserializer.Deserialize(inputStream);
			_singularValues = (float[]) deserializer.Deserialize(inputStream);
			_normalizeFactor =  (float) deserializer.Deserialize(inputStream);
			_thresholdRate =  (float) deserializer.Deserialize(inputStream);
			_isNormalizeOutput = (bool) deserializer.Deserialize(inputStream);

			inputStream.Close();

			_isPossibleNormalize = true;
		}

		public int InputVectorSize {
			get {
				if (!_isPossibleNormalize) {
					throw new Exception("Mean value dosn't calc");
				}
				return _normalizedInputVectorSize;
			}
		}

		public int OutputVectorSize {
			get { return _outputVectorSize; }
		}

		private void PrepareData(IList<TrainSingle> data) {
			_inputVectorSize = data[0].Input.Length;
			_outputVectorSize = 0;

			_inputMeanValueVector = new float[_inputVectorSize];
		}
		
		private void PrepareData(IList<TrainPair> data) {
			_inputVectorSize = data[0].Input.Length;
			_outputVectorSize = data[0].Output.Length;

			_inputMeanValueVector = new float[_inputVectorSize];
			_outputMeanValueVector = new float[_outputVectorSize];
		}

		private void CalcMeanValues(IList<TrainSingle> data, int dataSize) {
			var meanValueFactor = 1.0f/dataSize;
			for (var i = 0; i < dataSize; i++) {
				var inputVector = data[i].Input;
				for (var j = 0; j < _inputVectorSize; j++) {
					var value = inputVector[j];
					_inputMeanValueVector[j] += value*meanValueFactor;
				}
			}
		}

		private void CalcMeanValues(IList<TrainPair> data, int dataSize) {
			var meanValueFactor = 1.0f/dataSize;
			for (var i = 0; i < dataSize; i++) {
				var inputVector = data[i].Input;
				for (var j = 0; j < _inputVectorSize; j++) {
					var value = inputVector[j];
					_inputMeanValueVector[j] += value*meanValueFactor;
				}

				var outputVector = data[i].Output;
				for (var j = 0; j < _outputVectorSize; j++) {
					var value = outputVector[j];
					_outputMeanValueVector[j] += value*meanValueFactor;
				}
			}
		}

		private void SubtractMeanValues(IList<TrainSingle> data, int dataSize) {
			for (var i = 0; i < dataSize; i++) {
				var inputVector = data[i].Input;
				for (var j = 0; j < _inputVectorSize; j++) {
					inputVector[j] -= _inputMeanValueVector[j];
				}
			}
		}

		private void SubtractMeanValues(IList<TrainPair> data, int dataSize) {
			for (var i = 0; i < dataSize; i++) {
				var inputVector = data[i].Input;
				for (var j = 0; j < _inputVectorSize; j++) {
					inputVector[j] -= _inputMeanValueVector[j];
				}

				if (_isNormalizeOutput) {
					var outputVector = data[i].Output;
					for (var j = 0; j < _outputVectorSize; j++) {
						outputVector[j] -= _outputMeanValueVector[j];
					}
				}
			}
		}

		private void CreateFilter<T>(IList<T> data) where T:TrainSingle {
			var columnsList = new List<float[]>(data.Count);
		    columnsList.AddRange(data.Select(t => t.Input));
		    var matrix = DenseMatrix.OfColumns(_inputVectorSize, data.Count, columnsList);
			var svd = matrix.Svd(true);

			var fullTrasformMatrix = svd.U;
			var allSingularValues = svd.S;
			var minRealSingularValue = allSingularValues.Maximum()*_thresholdRate;
			_normalizedInputVectorSize = CalculateNumberRealSingularValues(allSingularValues, minRealSingularValue);
			_singularValues = new float[_normalizedInputVectorSize];
			_transformMatrix = new float[_inputVectorSize*_normalizedInputVectorSize];

			var realIndex = 0;
			for (var i = 0; i < allSingularValues.Count; i++) {
				if (allSingularValues[i] > minRealSingularValue) {
					_singularValues[realIndex] = allSingularValues[i];
					for (var j = 0; j < _inputVectorSize; j++) {
						_transformMatrix[realIndex*_inputVectorSize + j] = fullTrasformMatrix[j, i];
					}
					realIndex++;
				}
			}
		}

		private static int CalculateNumberRealSingularValues(IList<float> allSingularValues, float minSingularValue) {
			var size = allSingularValues.Count;
			var realValuesCount = 0;
			for (var i = 0; i < size; i++) {
				if (allSingularValues[i] > minSingularValue) {
					realValuesCount++;
				}
			}
			return realValuesCount;
		}

		private void ApplyFilterForInputComponents<T>(IList<T> data) where T:TrainSingle {
			_normalizeFactor = (float) Math.Sqrt(data.Count);
			for (var i = 0; i < data.Count; i++) {
				var input = data[i].Input;
				var normalizedInput = new float[_normalizedInputVectorSize];
				for (var j = 0; j < _normalizedInputVectorSize; j++) {
					var sum = 0.0f;
					for (var k = 0; k < _inputVectorSize; k++) {
						sum += input[k]*_transformMatrix[j*_inputVectorSize + k];
					}
					normalizedInput[j] = sum*_normalizeFactor/_singularValues[j];
				}
				data[i].Input = normalizedInput;
			}
		}
	}
}