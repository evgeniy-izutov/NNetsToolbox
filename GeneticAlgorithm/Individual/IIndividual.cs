namespace GeneticAlgorithm {
    public interface IIndividual {
        float[][] Chromosomes { get; }
        float Fitness { get; set; }
        bool IsFitnessAvailable { get; set; }
    }
}