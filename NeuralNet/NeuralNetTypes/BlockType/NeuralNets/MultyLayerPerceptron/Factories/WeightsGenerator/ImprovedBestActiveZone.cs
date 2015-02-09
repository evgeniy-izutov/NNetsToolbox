using System;
using System.Collections.Generic;
using StandardTypes;

namespace NeuralNet.MultyLayerPerceptron {
	internal delegate void FillVectorFunction(float factor, float[] vector);
	
	public sealed class ImprovedBestActiveZone : IMlpWeightGenerator {
		private readonly FillVectorFunction _fillVector;
		private readonly Random _uniformGenerator;
		private readonly List<TrainPair> _trainData;
		private readonly float _uniformFactor;

		public ImprovedBestActiveZone(Distribution distribution, List<TrainPair> trainData) {
			if (distribution == Distribution.Uniform) {
				_fillVector = FillUniformVector;
			}
			else {
				_fillVector = FillNormalVector;
			}
			_trainData = trainData;
			_uniformGenerator = new Random();
			_uniformFactor = (float) Math.Sqrt(3.0);
		}
		
		public void GenerateNewWeights(MultyLayerPerceptron perceptron) {
			var layers = perceptron.Layers;
			List<float[]> newData = null;

			var dataMaxSqrSum = FindMaxSqrSum(_trainData);
			FillLayer(layers[0], dataMaxSqrSum);
			if (layers.Length > 1) {
				newData = CalculateNewData(_trainData, layers[0]);
			}
			
			for (var layerNum = 1; layerNum < layers.Length - 1; layerNum++) {
				dataMaxSqrSum = FindMaxSqrSum(newData);
				FillLayer(layers[layerNum], dataMaxSqrSum);
				newData = CalculateNewData(newData, layers[layerNum]);
			}

			dataMaxSqrSum = FindMaxSqrSum(newData);
			FillLayer(layers[0], dataMaxSqrSum);
		}

		private static float FindMaxSqrSum(List<TrainPair> trainData) {
			var maxSqrSum = 0f;
			foreach (var example in trainData) {
				var sqrSum = 0f;
				var input = example.Input;
				for (var i = 0; i < input.Length; i++) {
					sqrSum += input[i]*input[i];
				}

				if (sqrSum > maxSqrSum) {
					maxSqrSum = sqrSum;
				}
			}
			return maxSqrSum;
		}

		private static float FindMaxSqrSum(List<float[]> data) {
			var maxSqrSum = 0f;
			foreach (var input in data) {
				var sqrSum = 0f;
				for (var i = 0; i < input.Length; i++) {
					sqrSum += input[i]*input[i];
				}

				if (sqrSum > maxSqrSum) {
					maxSqrSum = sqrSum;
				}
			}
			return maxSqrSum;
		}

		private void FillLayer(BaseNeuralBlock layer, float dataMaxSqrSum) {
			var weights = layer.GetWeights()[0];
			var bias = layer.GetBias();
			var prevLayerSize = weights.Length/layer.Size;
			var derivativeScale = layer.GetActivationFunction().GetMaxDerivativeZone(0.05f);
			var factor = derivativeScale/(float)Math.Sqrt((prevLayerSize + 1)*dataMaxSqrSum);

			_fillVector(factor, weights);
			_fillVector(factor, bias);
		}

		private void FillUniformVector(float factor, float[] vector) {
			factor *= _uniformFactor;
			for (var i = 0; i < vector.Length; i++) {
				vector[i] = factor*(2.0f*(float)_uniformGenerator.NextDouble() - 1.0f);
			}
		}

		private void FillNormalVector(float sigma, float[] vector) {
			float normal1, normal2;
			
			var length = vector.Length;
			if (length%2 != 0) {
				length--;
				GenerateStandardNormal(out normal1, out normal2);
				vector[length] = sigma*normal1;
			}
			for (var i = 0; i < length; i += 2) {
				GenerateStandardNormal(out normal1, out normal2);
				vector[i] = sigma*normal1;
				vector[i + 1] = sigma*normal2;
			}
		}

		private void GenerateStandardNormal(out float x1, out float x2) {
			double x, y;
			double s;
			do {
				x = 2.0*_uniformGenerator.NextDouble() - 1.0;
				y = 2.0*_uniformGenerator.NextDouble() - 1.0;
				s = x*x + y*y;
			} while ((s <= 0.0f) || (s > 1.0f));
			var factor = 1.0/s;
			factor = Math.Sqrt(2.0*factor*Math.Log(factor, Math.E));
			x1 = (float) (x*factor);
			x2 = (float) (y*factor);
		}

		private List<float[]> CalculateNewData(List<TrainPair> data, BaseNeuralBlock layer) {
			var result = new List<float[]>(data.Count);
			
			var weights = layer.GetWeights()[0];
			var bias = layer.GetBias();
			var prevLayerSize = weights.Length/layer.Size;
			
			foreach (var example in data) {
				var input = example.Input;
				var output = new float[layer.Size];
				for (var i = 0; i < layer.Size; i++) {
					var sum = bias[i];
					for (var j = 0; j < prevLayerSize; j++) {
						sum += input[j]*weights[i*prevLayerSize + j];
					}
					output[i] = sum;
				}
				result.Add(output);
			}

			return result;
		} 

		private List<float[]> CalculateNewData(List<float[]> data, BaseNeuralBlock layer) {
			var result = new List<float[]>(data.Count);
			
			var weights = layer.GetWeights()[0];
			var bias = layer.GetBias();
			var prevLayerSize = weights.Length/layer.Size;
			
			foreach (var input in data) {
				var output = new float[layer.Size];
				for (var i = 0; i < layer.Size; i++) {
					var sum = bias[i];
					for (var j = 0; j < prevLayerSize; j++) {
						sum += input[j]*weights[i*prevLayerSize + j];
					}
					output[i] = sum;
				}
				result.Add(output);
			}

			return result;
		} 
	}
}