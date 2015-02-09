using System;

namespace GeneticAlgorithm {
	public sealed class BlXalphaCrossoverWithBorder : ICrossoverOperator {
		private readonly float _alpha;
		private readonly float[] _minValue;
		private readonly float[] _maxValue;
		private readonly Random _random;

		public BlXalphaCrossoverWithBorder(float alpha, float[] minValue, float[] maxValue, int seed) {
			_alpha = alpha;
			_maxValue = maxValue;
			_minValue = minValue;
			_random = new Random(seed);
		}

		public void Cross(IIndividual parent1, IIndividual parent2, IIndividual child1, IIndividual child2) {
			var chromosomesParent1 = parent1.Chromosomes;
			var chromosomesParent2 = parent2.Chromosomes;
			var chromosomesChild1 = child1.Chromosomes;
			var chromosomesChild2 = child2.Chromosomes;
			var chromosomesCount = chromosomesChild1.Length;
			for (var i = 0; i < chromosomesCount; i++) {
				var chromosomeParent1 = chromosomesParent1[i];
				var chromosomeParent2 = chromosomesParent2[i];
				var chromosomeChild1 = chromosomesChild1[i];
				var chromosomeChild2 = chromosomesChild2[i];
				var chromosomeLength = chromosomeChild1.Length;
				for (var j = 0; j < chromosomeLength; j++) {
					var gene1 = chromosomeParent1[j]; //max = (a + b + |a - b|)*0.5
					var gene2 = chromosomeParent2[j]; //min = (a + b - |a - b|)*0.5
					var sumGenes = gene1 + gene2;
					var deltaFactor = (1.0f + 2.0f*_alpha)*Math.Abs(gene1 - gene2);

					var leftLimit = (sumGenes - deltaFactor)*0.5f;
					if (leftLimit < _minValue[i]) {
						leftLimit = _minValue[i];
					}
					var rightLimit = (sumGenes + deltaFactor)*0.5f;
					if (rightLimit > _maxValue[i]) {
						rightLimit = _maxValue[i];
					}
					var factor = rightLimit - leftLimit;
					var bias = leftLimit;

					chromosomeChild1[j] = factor*(float)_random.NextDouble() + bias;
					chromosomeChild2[j] = factor*(float)_random.NextDouble() + bias;
				}
			}
		}
	}
}