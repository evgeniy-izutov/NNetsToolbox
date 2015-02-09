namespace GeneticAlgorithm {
	public enum OptimizationCriterion {
		MinCriterion,
		MaxCriterion
	}
	
	public interface IInitilizeProperties {
		int PopulationsCount { get; set; }
		int PopulationSize { get; set; }
		float NewPopulationFactor { get; set; }
		float ElitePopulationFactor { get; set; }
		bool IsEliteIndividualAlways { get; set; }
		float CrossingProbability { get; set; }
		float MutationProbobility { get; set; }
		int IterationCount { get; set; }
		int[] ChromosomesStructure { get; set; }
		IFitnessFunction FitnessFunction { get; set; }
		ISelectionOperator SelectionOperator { get; set; }
		ICrossoverOperator CrossoverOperator { get; set; }
		IMutationOperator MutationOperator { get; set; }
		IChromosomesDistribution ChromosomesDistribution { get; set; }
		OptimizationCriterion Criterion { get; set; }
	}
}
