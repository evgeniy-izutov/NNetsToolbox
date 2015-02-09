using System;

namespace GeneticAlgorithm {
	public sealed class SingleMutation : IMutationOperator {
		private readonly float[] _factors;
		private readonly float[] _biases;
		private readonly Random _random;

		public SingleMutation(float[] minChromosomeValues, float[] maxChromosomeValues, int seed) {
			var length = minChromosomeValues.Length;
			_factors = new float[length];
			_biases = new float[length];
			for (var i = 0; i < length; i++) {
				_factors[i] = maxChromosomeValues[i] - minChromosomeValues[i];
				_biases[i] = minChromosomeValues[i];
			}
			_random = new Random(seed);
		}

		public void Mutate(IIndividual individual) {
			var chromosomes = individual.Chromosomes;
			for (var i = 0; i < chromosomes.Length; i++) {
				var chromosome = chromosomes[i];
				var positionForMutation = _random.Next(chromosome.Length);
				chromosome[positionForMutation] = _factors[i]*((float)_random.NextDouble()) + _biases[i];
			}
			individual.IsFitnessAvailable = false;
		}
	}
}