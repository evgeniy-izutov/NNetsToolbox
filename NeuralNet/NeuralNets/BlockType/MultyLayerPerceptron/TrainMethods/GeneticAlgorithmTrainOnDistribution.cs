using System;
using System.Collections.Generic;
using GeneticAlgorithm;
using StandardTypes;

namespace NeuralNet.MultyLayerPerceptron {
	public sealed class GeneticAlgorithmTrainOnDistribution : TrainMethod {
		private GeneticAlgorithm.GeneticAlgorithm _geneticAlgorithm;
        private MultyLayerPerceptron _neuralNet;
        private readonly IList<TrainPair> _trainingData;
		private readonly INormalizeMethod _normilazeMethod;
		private ITrainProperties _properties;
		private readonly float[] _distribution;
        private readonly int _populationSzie;
        private const int TournamentSize = 2;
        private const float Alpha = 0.6f;
        private const float NewPopulationFactor = 0.70f;
        private const float ElitePopulationFactor = 0.1f;
        private const float CrossingProbability = 0.9f;
        private const float MutationProbobility = 0.3f;
		
		public GeneticAlgorithmTrainOnDistribution (int populationSzie, 
			                                        INormalizeMethod normilazeMethod, 
			                                        float[] distribution,
			                                        IList<TrainPair> trainingData) {
			_populationSzie = populationSzie;
			_trainingData = trainingData;
			_distribution = distribution;
			_normilazeMethod = normilazeMethod;
		}

		public override void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties) {
			var net = neuralNet as MultyLayerPerceptron;
			if (net != null) {
                _neuralNet = net;
            }
            else {
                throw new ArgumentException("NeuronNet has other structure");
            }
			_properties = trainProperties;
            ProcessSate = IterativeProcessState.NotStarted;
		}

		public override ITrainProperties Properties {
			get { return _properties; }
		}

		protected override void RunIterativeProcess() {
			_geneticAlgorithm.Start();
		}

		public override void Stop() {
            if (_geneticAlgorithm != null) {
                _geneticAlgorithm.Stop();
            }
        }

		protected override void FirstRunInit() {
            _geneticAlgorithm = CreateGeneticAlgorithm(_neuralNet, _trainingData);
        }

        protected override void ApplyResults() {
            FinalChangeWeights(_geneticAlgorithm.GetResult(), _neuralNet);
        }

		private static void FinalChangeWeights(float[][] solution, MultyLayerPerceptron neuralNet) {
			var chromosomeIndex = 0;
			var layers = neuralNet.Layers;
			foreach (var neuronBlock in layers) {
				neuronBlock.SetWeightsFor(0, solution[chromosomeIndex++]);
				neuronBlock.SetBias(solution[chromosomeIndex++]);
			}
		}

		private GeneticAlgorithm.GeneticAlgorithm CreateGeneticAlgorithm (MultyLayerPerceptron neuralNet, IList<TrainPair> trainingData) {
            var random = new Random();
			int[] structure;
			float[] minValues;
			float[] maxValues;
			GetChromosomesStructure(neuralNet, out structure, out minValues, out maxValues);
			var initilizeProperties = new InitilizeProperties {
				PopulationsCount = 1,
				PopulationSize = _populationSzie,
				NewPopulationFactor = NewPopulationFactor,
				ElitePopulationFactor = ElitePopulationFactor,
				IsEliteIndividualAlways = true,
				CrossingProbability = CrossingProbability,
				MutationProbobility = MutationProbobility,
				IterationCount = _properties.MaxIterationCount,
				ChromosomesStructure = structure,
				FitnessFunction = new FitnessFunctionOnDistribution(neuralNet, trainingData, _normilazeMethod, _distribution),
				SelectionOperator = new TournamentSelection(TournamentSize, random.Next()),
				CrossoverOperator = new BlXalphaCrossoverWithoutBorder(Alpha, random.Next()),
				MutationOperator = new SingleMutation(minValues, maxValues, random.Next()),
				ChromosomesDistribution = new ChromosomesDistribution(DistributionType.Normal, minValues, maxValues, random.Next()),
				Criterion = OptimizationCriterion.MinCriterion
			};
			var geneticAlgorithm = new GeneticAlgorithm.GeneticAlgorithm();
			geneticAlgorithm.InitilazeAlgorithm(initilizeProperties, random.Next());
            geneticAlgorithm.IterationCompleted += GeneticAlgorithmIterationCopleted;
            geneticAlgorithm.IterativeProcessFinished += GeneticAlgorithmIterativeProcessFinished;
            return geneticAlgorithm;
        }

        private void GeneticAlgorithmIterationCopleted (object sender, IterationCompletedEventArgs e) {
            OnIterationCompleted(e);
        }

        private void GeneticAlgorithmIterativeProcessFinished (object sender, IterativeProcessFinishedEventArgs e) {
            OnIterativeProcessFinished(e);
        }

        private static void GetChromosomesStructure (MultyLayerPerceptron neuralNet, 
			                                         out int[] chromosomesStructure, 
			                                         out float[] minValues,
			                                         out float[] maxValues) {
            var layersStruct = neuralNet.GetLayersStruct();
        	var chromosomesCount = 2*(layersStruct.Length - 1);
			chromosomesStructure = new int[chromosomesCount];
			minValues = new float[chromosomesCount];
			maxValues = new float[chromosomesCount];
        	for (var i = 1; i < layersStruct.Length; i++) {
        		chromosomesStructure[2*(i - 1)] = layersStruct[i - 1]*layersStruct[i];
        		chromosomesStructure[2*(i - 1) + 1] = layersStruct[i];
				
				var border = (float)Math.Sqrt(1.0f/layersStruct[i - 1]);
        		minValues[2*(i - 1)] = -border;
				minValues[2*(i - 1) + 1] = -border;
				maxValues[2*(i - 1)] = border;
				maxValues[2*(i - 1) + 1] = border;
        	}
        }
	}
}
