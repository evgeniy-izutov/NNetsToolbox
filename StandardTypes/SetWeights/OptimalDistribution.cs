using System;
using System.Collections.Generic;
using System.Linq;

namespace StandardTypes.SetWeights {
	public sealed class OptimalDistribution<T> : ISetWeightsGenerator<T> where T:TrainPair {
		private const double K = 40d;
		private const double R = 2d;
		private const double AlphaStartValue = 0.1f;
		private const double BettaStartValue = 0.1f;
		private const float LowBorderValue = 0f;
		private	const float UpBorderValue = 1f;
		private const float EpsGradient = 0.00000001f;
		private const float EpsFunction = 0f;
		private const float EpsStep = 0f;
		private const float EpsOuter = 0.0000001f;
		private const float EpsInner = 0.0000001f;
		private const float DiffStep = 1.0e-6f;
		private List<float[]> _distributions;
		private int _examplesCount;
		private int _distributionSize;
		private float[] _weights;
		private float[] _exampleDistances;
		private float _averageDistance;
		private float[] _tempSums;
		private double[] _lowBorders;
		private double[] _upBorders;
		private double[] _x;

		public void GenerateWeights(List<T> set) {
			BuildDistributions(set);
			
			AllocateMemory();

			CalculateDistances();
			
			alglib.minbleicstate state;
			alglib.minbleiccreatef(_x, DiffStep, out state);
			alglib.minbleicsetbc(state, _lowBorders, _upBorders);
			alglib.minbleicsetinnercond(state, EpsGradient, EpsFunction, EpsStep);
			alglib.minbleicsetoutercond(state, EpsOuter, EpsInner);
			alglib.minbleicoptimize(state, DistanceToUniform, null, null);

			alglib.minbleicreport rep;
			alglib.minbleicresults(state, out _x, out rep);

			CalculateWeights(_weights, (float) _x[0], (float) _x[1], _averageDistance, _exampleDistances);

			SetWeights(_weights, set);
		}

		private void BuildDistributions(List<T> set) {
			_examplesCount = set.Count;

			_distributions = new List<float[]>(_examplesCount);
			for (var i = 0; i < _examplesCount; i++) {
				_distributions.Add(set[i].Output);
			}
			
			_distributionSize = _distributions.First().Length;
		} 

		private void AllocateMemory() {
			_weights = new float[_examplesCount];
			_exampleDistances = new float[_examplesCount];
			_tempSums = new float[_distributionSize];
			
			_lowBorders = new double[2];
			_upBorders = new double[2];
			for (var k = 0; k < 2; k++) {
				_lowBorders[k] = LowBorderValue;
				_upBorders[k] = UpBorderValue;
			}

			_x = new double[2];
			_x[0] = AlphaStartValue;
			_x[1] = BettaStartValue;
		}

		private void CalculateDistances() {
			var averageDistribution = CalculateAverageDistribution(_distributions);

			var uniformDistribution = BuildUniformDistribution(_distributionSize);
			_averageDistance = CalculateHellingerDistance(uniformDistribution, averageDistribution);

			for (var k = 0; k < _examplesCount; k++) {
				_exampleDistances[k] = CalculateHellingerDistance(averageDistribution, _distributions[k]);
			}
		}

		private static float[] CalculateAverageDistribution(List<float[]> distributions) {
			var examplesCount = distributions.Count;
			var distributionSize = distributions.First().Length;

			var averageDistribution = new float[distributionSize];

			var factor = 1f/examplesCount;
			for (var k = 0; k < examplesCount; k++) {
				var distribution = distributions[k];
				
				for (var i = 0; i < distributionSize; i++) {
					averageDistribution[i] += factor*distribution[i];
				}
			}

			return averageDistribution;
		}

		private static float[] BuildUniformDistribution(int size) {
			var distribution = new float[size];

			var value = 1f/size;
			for (var i = 0; i < size; i++) {
				distribution[i] = value;
			}

			return distribution;
		}

		private void DistanceToUniform(double[] x, ref double funcValue, object obj) {
			CalculateWeights(_weights, (float) x[0], (float) x[1], _averageDistance, _exampleDistances);

			for (var i = 0; i < _distributionSize; i++) {
				_tempSums[i] = 0f;
			}
			
			for (var k = 0; k < _examplesCount; k++) {
				var weight = _weights[k];
				var distribution = _distributions[k];
				
				for (var i = 0; i < _distributionSize; i++) {
					_tempSums[i] += weight*distribution[i];
				}
			}
			
			funcValue = 0f;
			for (var i = 0; i < _distributionSize; i++) {
				funcValue -= (float) Math.Sqrt(_tempSums[i]/_examplesCount);
			}

			funcValue += 1f;
		}

		private static void CalculateWeights(float[] weights, float alpha, float betta,
											 float averageDistance, float[] exampleDistances) {
			var sumWeights = 0f;
			for (var k = 0; k < weights.Length; k++) {
				var weight = (float) Math.Exp(R*Math.Tanh(K*(averageDistance - alpha))
					*Math.Tanh(K*(exampleDistances[k] - betta)));

				weights[k] = weight;
				sumWeights += weight;
			}

			for (var k = 0; k < weights.Length; k++) {
				weights[k] /= sumWeights;
			}
		}

		private static float CalculateHellingerDistance(float[] real, float[] empirical) {
			var sum = 0f;

			for (var i = 0; i < real.Length; i++) {
				sum += (float) Math.Sqrt(real[i]*empirical[i]);
			}

			return 1f*(1f - sum);
		}

		private static void SetWeights(float[] weights, List<T> set) {
			for (var i = 0; i < weights.Length; i++) {
				set[i].Weight = weights[i];
			}
		}
	}
}
