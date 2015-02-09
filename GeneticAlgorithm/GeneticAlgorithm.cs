using System;
using System.Collections.Generic;
using StandardTypes;

namespace GeneticAlgorithm {
	public sealed class GeneticAlgorithm : IterativeProcess {
        private Random _random;
		private IPopulation _population;
		private IPopulation _bufferPopulation;
    	private IIndividual[] _matingPool;
		private Queue<IIndividual> _individualsBuffer;
        private IFitnessFunction _fitnessFunction;
        private ISelectionOperator _selectionOperator;
        private ICrossoverOperator _crossoverOperator;
        private IMutationOperator _mutationOperator;
    	private IChromosomesDistribution _chromosomesDistribution;
    	private int[] _chromosomesStructure;
        private int _populationSize;
        private float _crossingProbability;
        private float _mutationProbobility;
        private int _newPopulationSize;
        private int _oldPopulationSize;
        private int _elitePopulationSize;
		private int _iterCount;
		private IBestFitness _bestFitness;
		private OptimizationCriterion _criterion;

        public GeneticAlgorithm() {
        }

        public void InitilazeAlgorithm(IInitilizeProperties properties, int seed) {
        	_chromosomesStructure = properties.ChromosomesStructure;
			_populationSize = properties.PopulationSize;
            _mutationProbobility = properties.MutationProbobility;
            _crossingProbability = properties.CrossingProbability;
            _newPopulationSize = (int)(properties.NewPopulationFactor*_populationSize);
            if (_newPopulationSize%2 != 0) {
                _newPopulationSize--;
            }
			_oldPopulationSize = _populationSize - _newPopulationSize;
            _elitePopulationSize = (int)(properties.ElitePopulationFactor*_oldPopulationSize);
            if ((_elitePopulationSize == 0) && properties.IsEliteIndividualAlways && (_oldPopulationSize != 0)) {
                _elitePopulationSize = 1;
            }
            
			_individualsBuffer = new Queue<IIndividual>(_newPopulationSize);
        	for (var i = 0; i < _newPopulationSize; i++) {
        		_individualsBuffer.Enqueue(new Individual(_chromosomesStructure));
        	}
			_bufferPopulation = new Population(_populationSize);
			_matingPool = new IIndividual[_newPopulationSize];
            _random = new Random(seed);
			_iterCount = properties.IterationCount;
        	_fitnessFunction = properties.FitnessFunction;
        	_selectionOperator = properties.SelectionOperator;
        	_crossoverOperator = properties.CrossoverOperator;
        	_mutationOperator = properties.MutationOperator;
        	_chromosomesDistribution = properties.ChromosomesDistribution;
        	_criterion = properties.Criterion;

            ProcessSate = IterativeProcessState.NotStarted;
        }

        public float[][] GetResult() {
            if (ProcessSate < IterativeProcessState.Stoped) {
                throw new ApplicationException("Result data is not ready yet");
            }
            return _population[_bestFitness.Position].Chromosomes;
        }

        protected override void FirstRunInit() {
            CreateStartPopulation();
        }

        protected override void RunIterativeProcess() {
            var iteration = 0;
            while ((ProcessSate == IterativeProcessState.InProgress) && (iteration < _iterCount)) {
                SortPopulationByFitnessValue();
                BuildNewPopulation();
                OnIterationCompleted(new IterationCompletedEventArgs(iteration, _bestFitness.Value, 0.0f));
                iteration++;
            }
            OnIterativeProcessFinished(new IterativeProcessFinishedEventArgs(iteration));
        }
        
        private void CreateStartPopulation() {
            _population = new Population(_populationSize, _chromosomesStructure, _chromosomesDistribution);
            CalcFitnessValues(_population);
        }

        private void BuildNewPopulation() {
        	_bufferPopulation.Reset();
            AddCrossingIndividuals(_bufferPopulation);
			AddEliteAndOldIndividuals(_bufferPopulation);
            MutatePopulation(_bufferPopulation, _newPopulationSize + _elitePopulationSize, _populationSize - 1);
            CalcFitnessValues(_bufferPopulation);

        	var tmp = _population;
            _population = _bufferPopulation;
        	_bufferPopulation = tmp;
        }

        private void CalcFitnessValues(IPopulation population) {
            float bestFitnessValue;
            var fitnessSum = 0.0f;
            var bestFitnessPos = 0;
            var size = population.Size;

        	if (_criterion == OptimizationCriterion.MinCriterion) {
        		bestFitnessValue = Single.MaxValue;
				for (var i = 0; i < size; i++) {
					var individual = population[i];
					if (!individual.IsFitnessAvailable) {
						_fitnessFunction.Fitness(individual);
					}
					var individualFitnessValue = individual.Fitness;
					fitnessSum += individualFitnessValue;
					if (individualFitnessValue < bestFitnessValue) {
						bestFitnessValue = individualFitnessValue;
						bestFitnessPos = i;
					}
				}
        	}
        	else {
        		bestFitnessValue = Single.MinValue;
				for (var i = 0; i < size; i++) {
					var individual = population[i];
					if (!individual.IsFitnessAvailable) {
						_fitnessFunction.Fitness(individual);
					}
					var individualFitnessValue = individual.Fitness;
					fitnessSum += individualFitnessValue;
					if (individualFitnessValue > bestFitnessValue) {
						bestFitnessValue = individualFitnessValue;
						bestFitnessPos = i;
					}
				}
        	}

            _bestFitness = new BestFitness(bestFitnessValue, bestFitnessPos, fitnessSum);
        }

        private void SortPopulationByFitnessValue() {
        	if (_criterion == OptimizationCriterion.MinCriterion) {
        		QuickSortForMin(0, _population.Size - 1);
        	}
        	else {
        		QuickSortForMax(0, _population.Size - 1);
        	}
            _bestFitness.Position = 0;
        }

        private void QuickSortForMin(int first, int last) {
            var left = first;
            var right = last;
            var v = _population[(left + right)/2].Fitness;
            while (left <= right) {
                while (_population[left].Fitness < v) {
                    left++;
                }
                while (_population[right].Fitness > v) {
                    right--;
                }
                if (left <= right) {
                    var tmp = _population[left];
                    _population[left] = _population[right];
                    _population[right] = tmp;
                    left++;
                    right--;
                }
            }
            if (first < right) {
                QuickSortForMin(first, right);
            }
            if (left < last) {
                QuickSortForMin(left, last);
            }
        }

		private void QuickSortForMax(int first, int last) {
            var left = first;
            var right = last;
            var v = _population[(left + right)/2].Fitness;
            while (left <= right) {
                while (_population[left].Fitness > v) {
                    left++;
                }
                while (_population[right].Fitness < v) {
                    right--;
                }
                if (left <= right) {
                    var tmp = _population[left];
                    _population[left] = _population[right];
                    _population[right] = tmp;
                    left++;
                    right--;
                }
            }
            if (first < right) {
                QuickSortForMax(first, right);
            }
            if (left < last) {
                QuickSortForMax(left, last);
            }
        }

        private void AddCrossingIndividuals(IPopulation population) {
            _selectionOperator.Select(_matingPool, _population, _bestFitness, _criterion);
            var curIndividualsCount = 0;
            while (curIndividualsCount < _newPopulationSize) {
                if (_random.NextDouble() < _crossingProbability) {
                    var firstParent = _random.Next(_newPopulationSize);
                    var secondParent = _random.Next(_newPopulationSize);
                    var firstChild = _individualsBuffer.Dequeue();
                    var secondChild = _individualsBuffer.Dequeue();
                    _crossoverOperator.Cross(_matingPool[firstParent], _matingPool[secondParent], firstChild, secondChild);
                    population.AddIndividual(firstChild);
                    population.AddIndividual(secondChild);
                    curIndividualsCount += 2; 
                }
            }
        }

        private void AddEliteAndOldIndividuals(IPopulation population) {
			for (var i = 0; i < _elitePopulationSize; i++) {
                population.AddIndividual(_population[i]);
            }

        	var startPos = _elitePopulationSize;
        	for (var i = 0; i < _oldPopulationSize - _elitePopulationSize; i++) {
        		var index = _random.Next(startPos, _populationSize);
				if (index != startPos) {
					var tmp = _population[startPos];
					_population[startPos] = _population[index];
					_population[index] = tmp;
				}
				population.AddIndividual(_population[startPos]);
        		startPos++;
        	}

        	for (var i = _oldPopulationSize; i < _populationSize; i++) {
        		var bufferIndividual = _population[i];
        		bufferIndividual.IsFitnessAvailable = false;
				_individualsBuffer.Enqueue(bufferIndividual);
        	}
        }

        private void MutatePopulation(IPopulation population, int leftBorder, int rightBorder) {
            for (var i = leftBorder; i <= rightBorder; i++) {
                if (_random.NextDouble() < _mutationProbobility) {
                    _mutationOperator.Mutate(population[i]);
                }
            }
        }
    }
}