namespace GeneticAlgorithm {
	public sealed class InitilizeProperties : IInitilizeProperties {
		public int PopulationsCount { get; set; }
		public int PopulationSize { get; set; }
		public float NewPopulationFactor { get; set; }
		public float ElitePopulationFactor { get; set; }
		public bool IsEliteIndividualAlways { get; set; }
		public float CrossingProbability { get; set; }
		public float MutationProbobility { get; set; }
		public int IterationCount { get; set; }
		public int[] ChromosomesStructure { get; set; }
		public IFitnessFunction FitnessFunction { get; set; }
		public ISelectionOperator SelectionOperator { get; set; }
		public ICrossoverOperator CrossoverOperator { get; set; }
		public IMutationOperator MutationOperator { get; set; }
		public IChromosomesDistribution ChromosomesDistribution { get; set; }
		public OptimizationCriterion Criterion { get; set; }
	}
}
