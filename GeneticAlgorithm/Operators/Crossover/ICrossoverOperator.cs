namespace GeneticAlgorithm {
    public interface ICrossoverOperator {
        void Cross(IIndividual parent1, IIndividual parent2, IIndividual child1, IIndividual child2);
    }
}