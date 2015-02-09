namespace GeneticAlgorithm {
    public interface ISelectionOperator {
        void Select(IIndividual[] matingPool, IPopulation population, IBestFitness bestFitness, OptimizationCriterion criterion);
    }
}