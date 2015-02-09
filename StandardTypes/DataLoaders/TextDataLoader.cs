using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace StandardTypes {
	public sealed class TextDataLoader : IDataLoader {
		private readonly string _sourceFilePath;
		private readonly string _missedSymbol;
		private readonly float _missedValue;
		private readonly char _featureSeparator;
		private readonly float _trueValue;
		private readonly float _falsevalue;
		private readonly NumberFormatInfo _numberFormat;
		
		public TextDataLoader(string sourceFilePath, string missedSymbol, float missedValue, string decimalSeparator, char featureSeparator, float trueValue, float falsevalue) {
			_sourceFilePath = sourceFilePath;
			_missedSymbol = missedSymbol;
			_missedValue = missedValue;
			_featureSeparator = featureSeparator;
			_trueValue = trueValue;
			_falsevalue = falsevalue;

			_numberFormat = new CultureInfo( "en-US", false ).NumberFormat;
			_numberFormat.NumberDecimalSeparator = decimalSeparator;
		}

		public List<TrainSingle> LoadData(List<FeatureDescription> inputDescription) {
			var data = new List<TrainSingle>();
			var inputSize = CalculateVectorSize(inputDescription);

			var inputStream = new StreamReader(_sourceFilePath);
			while (!inputStream.EndOfStream) {
				var line = inputStream.ReadLine();
				if (!string.IsNullOrEmpty(line)) {
					var lineParts = line.Split(_featureSeparator);
					var example = CreateSingle(lineParts, inputDescription, inputSize);
					data.Add(example);
				}
			}
			inputStream.Close();
			
			return data;
		}

		public List<TrainPair> LoadData(List<FeatureDescription> inputDescription, List<FeatureDescription> outputDescription) {
			var data = new List<TrainPair>();
			var inputSize = CalculateVectorSize(inputDescription);
			var outputSize = CalculateVectorSize(outputDescription);

			var inputStream = new StreamReader(_sourceFilePath);
			while (!inputStream.EndOfStream) {
				var line = inputStream.ReadLine();
				if (!string.IsNullOrEmpty(line)) {
					var lineParts = line.Split(_featureSeparator);
					var example = CreatePair(lineParts, inputDescription, inputSize, outputDescription, outputSize);
					data.Add(example);
				}
			}
			inputStream.Close();
			
			return data;
		}

		private TrainSingle CreateSingle(string[] features, List<FeatureDescription> inputFeaturesDescription, int inputSize) {
			int exampleId;
			float exampleWeight;
			float[] input;
			HashSet<int> inputMissedIndexes;

			CompleteVector(inputSize, inputFeaturesDescription, features, out input, out inputMissedIndexes);
			CompleteIdAndWeight(inputFeaturesDescription, features, out exampleId, out exampleWeight);
			var example = new TrainSingle(input, inputMissedIndexes);
			example.Id = exampleId;
			example.Weight = exampleWeight;
			return example;
		}

		private TrainPair CreatePair(string[] features, List<FeatureDescription> inputFeaturesDescription, int inputSize, List<FeatureDescription> outputFeaturesDescription, int outputSize) {
			int exampleId;
			float exampleWeight;
			float[] input, output;
			HashSet<int> inputMissedIndexes, outputMissedIndexes;

			CompleteVector(inputSize, inputFeaturesDescription, features, out input, out inputMissedIndexes);
			CompleteVector(outputSize, outputFeaturesDescription, features, out output, out outputMissedIndexes);
			CompleteIdAndWeight(inputFeaturesDescription, features, out exampleId, out exampleWeight);
			var example = new TrainPair(input, output, inputMissedIndexes, outputMissedIndexes);
			example.Id = exampleId;
			example.Weight = exampleWeight;
			return example;
		}

		private static int CalculateVectorSize(List<FeatureDescription> featuresDescription) {
			var size = 0;
			foreach (var feature in featuresDescription) {
				if (feature.Type == FeatureType.Value) {
					size++;
				}
				else if (feature.Type == FeatureType.Categorial) {
					size += feature.CategoriesCount;
				}
			}
			return size;
		}

		private void CompleteVector(int vectorSize, List<FeatureDescription> featuresDescription, string[] features, 
			out float[] vector, out HashSet<int> missedIndexes) {
			vector = new float[vectorSize];
			missedIndexes = null;

			var index = 0;
			foreach (var featureDescription in featuresDescription) {
				var value = features[featureDescription.SourceId];

				switch (featureDescription.Type) {
					case FeatureType.Value:
						if (value == _missedSymbol) {
							if (missedIndexes == null) {
								missedIndexes = new HashSet<int>();
							}
							missedIndexes.Add(index);
							vector[index] = _missedValue;
							index++;
						}
						else {
							vector[index] = (float) Convert.ToDouble(value, _numberFormat);
							index++;
						}
						break;
					case FeatureType.Categorial:
						if (value == _missedSymbol) {
							if (missedIndexes == null) {
								missedIndexes = new HashSet<int>();
							}
							for (var i = 0; i < featureDescription.CategoriesCount; i++) {
								missedIndexes.Add(index);
								vector[index] = _missedValue;
								index++;
							}
						}
						else {
							var position = (Int32) Convert.ToDouble(value, _numberFormat);
							for (var i = 0; i < position; i++) {
								vector[index] = _falsevalue;
								index++;
							}
							vector[index++] = _trueValue;
							for (var i = position + 1; i < featureDescription.CategoriesCount; i++) {
								vector[index] = _falsevalue;
								index++;
							}
						}
						break;
				}
			}
		}

		private void CompleteIdAndWeight(List<FeatureDescription> featuresDescription, string[] features, out int exampleId, out float exampleWeight) {
			var localId = -1;
			var localWeight = 1.0f;
			foreach (var featureDescription in featuresDescription) {
				if (featureDescription.Type == FeatureType.Id) {
					localId = Convert.ToInt32(features[featureDescription.SourceId]);
				}
				else if (featureDescription.Type == FeatureType.Weight) {
					localWeight = (float) Convert.ToDouble(features[featureDescription.SourceId], _numberFormat);
				}
			}
			exampleId = localId;
			exampleWeight = localWeight;
		}
	}
}