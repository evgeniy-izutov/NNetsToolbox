namespace GeneticAlgorithm {
    public interface IBestFitness {
        float Value { get; set; }
        int Position { get; set; }
        float TotalSum { get; set; }
    }
}