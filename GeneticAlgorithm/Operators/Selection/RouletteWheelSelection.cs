using System;

namespace GeneticAlgorithm {
	public sealed class RouletteWheelSelection : ISelectionOperator {
		private float[] _rouletteSegments;
		private readonly Random _random;

		public RouletteWheelSelection(int seed) {
			_random = new Random(seed);
		}

		public void Select(IIndividual[] matingPool, IPopulation population, IBestFitness bestFitness, OptimizationCriterion criterion) {
			var matingPoolSize = matingPool.Length;
			CreateRouletteSegments(matingPoolSize, population, bestFitness, criterion);
			for (var i = 0; i < matingPoolSize; i++) {
				var number = FindIndividualNumber((float)_random.NextDouble());
				matingPool[i] = population[number];
			}
		}

		private void CreateRouletteSegments (int matingPoolSize, IPopulation population, IBestFitness bestFitness, OptimizationCriterion criterion) {
			_rouletteSegments = new float[matingPoolSize];
			if (criterion == OptimizationCriterion.MaxCriterion) {
				var fitnessSum = bestFitness.TotalSum;
				_rouletteSegments[0] = population[0].Fitness/fitnessSum;
				for (var i = 1; i < matingPoolSize - 1; i++) {
					_rouletteSegments[i] = _rouletteSegments[i - 1] + population[i].Fitness/fitnessSum;
				}
				_rouletteSegments[matingPoolSize - 1] = 1.0f;
			}
			else {
				throw new NotImplementedException("Very slow for implementing.");
			}
		}

		private int FindIndividualNumber (float rouletteValue) {
			if (rouletteValue <= _rouletteSegments[0]) {
				return 0;
			}

			var left = 1;
			var right = _rouletteSegments.Length - 1;
			var index = (left + right)/2;
			while (left <= right) {
				if ((rouletteValue > _rouletteSegments[index - 1]) && (rouletteValue <= _rouletteSegments[index])){
					break;
				}
				if (rouletteValue < _rouletteSegments[index]) {
					right = index - 1;
				}
				else {
					left = index + 1;
				}
				index = (left + right)/2;
			}
			return index;
		}
	}
}