using System;

namespace GeneticAlgorithm {
	internal sealed class Individual : IIndividual {
		private readonly float[][] _chromosomes;
		private float _fitness;
		//private static readonly float Sqrt12 = (float)Math.Sqrt(12.0);

		public Individual (int[] chromosomesStructure) {
			var chromosomesCount = chromosomesStructure.Length;
			_chromosomes = new float[chromosomesCount][];
			for (var i = 0; i < chromosomesCount; i++) {
				_chromosomes[i] = new float[chromosomesStructure[i]];
			}

			//switch (distributionLaw) {
			//    case DistributionLaw.Evenly:
			//        var factor = standardDeviation*Sqrt12;
			//        var bias = meanValue - 0.5f*factor;
			//        for (var i = 0; i < genotypeLength; i++) {
			//            _genotype[i] = factor*(float)random.NextDouble() + bias;
			//        }
			//        break;
			//    case DistributionLaw.Normal:
			//        float s, v1, v2, normalValue;
			//        for (var i = 0; i < genotypeLength; i++) {
			//            do {
			//                v1 = 2.0f*(float)random.NextDouble() - 1.0f;
			//                v2 = 2.0f*(float)random.NextDouble() - 1.0f;
			//                s = v1*v1 + v2*v2;
			//            } while (s >= 1.0);
			//            normalValue = v1*(float)Math.Sqrt(-2.0*Math.Log(s, Math.E)/s);
			//            _genotype[i] = standardDeviation*normalValue + meanValue;
			//        }
			//        break;
			//}

			IsFitnessAvailable = false;
		}
        
		public Individual (float[][] chromosomes) {
			_chromosomes = chromosomes;
			IsFitnessAvailable = false;
		}

		public Individual (IIndividual individual) {
			var sourceChromosomes = individual.Chromosomes;
			var chromosomesCount = sourceChromosomes.Length;
			_chromosomes = new float[chromosomesCount][];
			for (var i = 0; i < chromosomesCount; i++) {
				var sourceChromosome = sourceChromosomes[i];
				var chromosomeLength = sourceChromosome.Length;
				var newChromosome = new float[chromosomeLength];
				for (var j = 0; j < chromosomeLength; j++) {
					newChromosome[j] = sourceChromosome[j];
				}
				_chromosomes[i] = newChromosome;
			}
			_fitness = individual.Fitness;
			IsFitnessAvailable = individual.IsFitnessAvailable;
		}

		public float[][] Chromosomes {
			get {
				return _chromosomes;
			}
		}

		public float Fitness {
			get {
				return _fitness;
			}
			set {
				_fitness = value;
				IsFitnessAvailable = true;
			}
		}

		public bool IsFitnessAvailable { get; set; }
	}
}