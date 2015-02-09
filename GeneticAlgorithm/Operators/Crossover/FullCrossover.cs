namespace GeneticAlgorithm {
	public sealed class FullCrossover : ICrossoverOperator {
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
				for (var j = 0; j < chromosomeLength - 1; j += 2) {
					chromosomeChild1[j] = chromosomeParent1[j];
					chromosomeChild2[j] = chromosomeParent2[j];

					chromosomeChild1[j + 1] = chromosomeParent2[j + 1];
					chromosomeChild2[j + 1] = chromosomeParent1[j + 1];
				}
				if (chromosomeLength%2 == 1) {
					var pos = chromosomeLength - 1;
					chromosomeChild1[pos] = chromosomeParent1[pos];
					chromosomeChild2[pos] = chromosomeParent2[pos];
				}
			}
		}
	}
}