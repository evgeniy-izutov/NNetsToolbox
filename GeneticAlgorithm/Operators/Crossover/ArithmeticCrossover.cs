using System;

namespace GeneticAlgorithm {
	public sealed class ArithmeticCrossover : ICrossoverOperator {
		private readonly Random _random;

		public ArithmeticCrossover(int seed) {
			_random = new Random(seed);
		}

		public void Cross(IIndividual parent1, IIndividual parent2, IIndividual child1, IIndividual child2) {
			var alpha = (float)_random.NextDouble();
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
					chromosomeChild1[j] = alpha*chromosomeParent1[j] + (1.0f - alpha)*chromosomeParent2[j];
					chromosomeChild2[j] = alpha*chromosomeParent2[j] + (1.0f - alpha)*chromosomeParent1[j];
				}
			}
		}
	}
}