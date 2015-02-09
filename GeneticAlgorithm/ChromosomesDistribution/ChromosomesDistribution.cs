using System;
using StandardTypes;

namespace GeneticAlgorithm {
	public sealed class ChromosomesDistribution : IChromosomesDistribution {
		private readonly Random _random;
		private readonly DistributionType _distributionType;
		private readonly float[] _minChromosomeValues;
		private readonly float[] _maxChromosomeValues;

		public ChromosomesDistribution(DistributionType startWeightGenerator, float[] minChromosomeValues, float[] maxChromosomeValues, int seed) {
			_random = new Random(seed);
			_distributionType = startWeightGenerator;
			_minChromosomeValues = minChromosomeValues;
			_maxChromosomeValues = maxChromosomeValues;
		}
		
		public void Initilize(float[][] chromosomes) {
			var chromosomesCount = chromosomes.Length;
			switch (_distributionType) {
				case DistributionType.Null:
					break;
				case DistributionType.Uniform:
					for (var i = 0; i < chromosomesCount; i++) {
						var chromosome = chromosomes[i];
						var chromosomeLength = chromosome.Length;
						var factor = _maxChromosomeValues[i] - _minChromosomeValues[i];
						var bias = _minChromosomeValues[i];
						for (var j = 0; j < chromosomeLength; j++) {
							chromosome[j] = factor*((float) _random.NextDouble()) + bias;
						}
					}
					break;
				case DistributionType.Normal:
					for (var i = 0; i < chromosomesCount; i++) {
						var chromosome = chromosomes[i];
						var chromosomeLength = chromosome.Length;
						var factor = (_maxChromosomeValues[i] - _minChromosomeValues[i])*0.5f;
						var bias = (_maxChromosomeValues[i] + _minChromosomeValues[i])*0.5f;

						float normal1, normal2;
						var length = chromosomeLength;
						if (length%2 != 0) {
							length--;
							GenerateNormal(_random, out normal1, out normal2);
							chromosome[length] = factor*normal1 + bias;
						}
						for (var j = 0; j < length; j += 2) {
							GenerateNormal(_random, out normal1, out normal2);
							chromosome[j] = factor*normal1 + bias;
							chromosome[j + 1] = factor*normal2 + bias;
						}
					}
					break;
			}
		}

		private static void GenerateNormal(Random random, out float x1, out float x2) {
			double x, y;
			double s;
			do {
				x = 2.0*random.NextDouble() - 1.0;
				y = 2.0*random.NextDouble() - 1.0;
				s = x*x + y*y;
			} while ((s <= 0.0f) || (s > 1.0f));
			var factor = 1.0/s;
			factor = Math.Sqrt(2.0*factor*Math.Log(factor, Math.E));
			x1 = (float) (x*factor);
			x2 = (float) (y*factor);
		}
	}
}
