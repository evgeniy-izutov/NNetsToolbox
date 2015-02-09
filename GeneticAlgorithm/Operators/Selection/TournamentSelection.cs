using System;

namespace GeneticAlgorithm {
	public sealed class TournamentSelection : ISelectionOperator {
		private readonly int _tournamentSize;
		private readonly Random _random;

		public TournamentSelection(int tournamentSize, int seed) {
			_tournamentSize = tournamentSize;
			_random = new Random(seed);
		}

		public void Select(IIndividual[] matingPool, IPopulation population, IBestFitness bestFitness, OptimizationCriterion criterion) {
			var populationSize = population.Size;
			var matingPoolSize = matingPool.Length;

			if (criterion == OptimizationCriterion.MinCriterion) {
				for (var i = 0; i < matingPoolSize; i++) {
					var minFitness = Single.MaxValue;
					var winnerNum = 0;
					for (var j = 0; j < _tournamentSize; j++) {
						var position = _random.Next(populationSize);
						var player = population[position];
						var fitnessValueOfPlayer = player.Fitness;
						if (fitnessValueOfPlayer < minFitness) {
							minFitness = fitnessValueOfPlayer;
							winnerNum = position;
						}
					}
					matingPool[i] = population[winnerNum];
				}
			}
			else {
				for (var i = 0; i < matingPoolSize; i++) {
					var maxFitness = Single.MinValue;
					var winnerNum = 0;
					for (var j = 0; j < _tournamentSize; j++) {
						var position = _random.Next(populationSize);
						var player = population[position];
						var fitnessValueOfPlayer = player.Fitness;
						if (fitnessValueOfPlayer > maxFitness) {
							maxFitness = fitnessValueOfPlayer;
							winnerNum = position;
						}
					}
					matingPool[i] = population[winnerNum];
				}
			}
		}
	}
}