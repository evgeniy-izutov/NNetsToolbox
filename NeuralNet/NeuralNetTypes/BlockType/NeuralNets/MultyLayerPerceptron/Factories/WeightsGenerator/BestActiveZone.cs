using System;

namespace NeuralNet.MultyLayerPerceptron {
	public sealed class BestActiveZone : IMlpWeightGenerator {
		private readonly Distribution _distribution;
		private readonly Random _uniformGenerator;

		public BestActiveZone(Distribution distribution) {
			_distribution = distribution;
			_uniformGenerator = new Random();
		}

		public void GenerateNewWeights(MultyLayerPerceptron perceptron) {
			var layers = perceptron.Layers;
			switch (_distribution) {
				case Distribution.Uniform:
					for (var layerNum = 0; layerNum < layers.Length; layerNum++) {
						var weights = layers[layerNum].GetWeights()[0];
						var bias = layers[layerNum].GetBias();
						var prevLayerSize = weights.Length/layers[layerNum].Size;
						var derivativeScale = layers[layerNum].GetActivationFunction().GetMaxDerivativeZone(0.05f);
						var factor = derivativeScale/(float)Math.Sqrt(3.0/(prevLayerSize + 1));
						
						FillUniformVector(factor, weights);
						FillUniformVector(factor, bias);
					}
					break;
				case Distribution.Normal:
					for (var layerNum = 0; layerNum < layers.Length; layerNum++) {
						var weights = layers[layerNum].GetWeights()[0];
						var bias = layers[layerNum].GetBias();
						var prevLayerSize = weights.Length / layers[layerNum].Size;
						var derivativeScale = layers[layerNum].GetActivationFunction().GetMaxDerivativeZone(0.05f);
						var sigma = derivativeScale/(float)Math.Sqrt(prevLayerSize + 1);
						
						FillNormalVector(sigma, 0.0f, weights);
						FillNormalVector(sigma, 0.0f,  bias);
					}
					break;
			}
		}

		private void FillUniformVector(float factor, float[] vector) {
			for (var i = 0; i < vector.Length; i++) {
				vector[i] = factor*(2.0f*(float)_uniformGenerator.NextDouble() - 1.0f);
			}
		}

		private void FillNormalVector(float sigma, float mean, float[] vector) {
			float normal1, normal2;
			
			var length = vector.Length;
			if (length%2 != 0) {
				length--;
				GenerateStandardNormal(out normal1, out normal2);
				vector[length] = sigma*normal1 + mean;
			}
			for (var i = 0; i < length; i += 2) {
				GenerateStandardNormal(out normal1, out normal2);
				vector[i] = sigma*normal1 + mean;
				vector[i + 1] = sigma*normal2 + mean;
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
	}
}