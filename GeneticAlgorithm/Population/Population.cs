namespace GeneticAlgorithm {
	internal sealed class Population : IPopulation {
		private IIndividual[] _individuals;
		private int _topIndex;

		public Population (int populationSize) {
			AllocateMemory(populationSize);
			_topIndex = 0;
		}

		public Population (int populationSize, int[] chromosomesStructure, IChromosomesDistribution chromosomesDistribution) : this(populationSize) {
			for (var i = 0; i < populationSize; i++) {
				var newIndividual = new Individual(chromosomesStructure);
				_individuals[_topIndex++] = newIndividual;
				chromosomesDistribution.Initilize(newIndividual.Chromosomes);
			}
		}

		public void AddIndividual (IIndividual individual) {
			_individuals[_topIndex++] = individual;
		}

		public void Reset() {
			_topIndex = 0;
		}

		public IIndividual this[int index] {
			get {
				return _individuals[index];
			}
			set {
				_individuals[index] = value;
			}
		}

		public int Size {
			get {
				return _topIndex;
			}
		}

		private void AllocateMemory (int size) {
			_individuals = new IIndividual[size];
		}
	}
}