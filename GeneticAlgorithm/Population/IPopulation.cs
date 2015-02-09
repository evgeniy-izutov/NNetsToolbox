namespace GeneticAlgorithm {
    public interface IPopulation {
        IIndividual this[int index] { get; set; }
        int Size { get; }
        void AddIndividual(IIndividual individual);
    	void Reset();
    }
}