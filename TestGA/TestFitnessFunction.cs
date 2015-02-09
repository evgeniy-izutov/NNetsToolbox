using System;
using GeneticAlgorithm;

namespace TestGA {
    class TestFitnessFunction : IFitnessFunction {
		public void Fitness(IIndividual individual) {
        	var chromosome = individual.Chromosomes[0];
            var fitnessValue = 0.0;
            for (var i = 0; i < chromosome.Length; i++) {
                var x = chromosome[i];
                fitnessValue -= x*Math.Sin(Math.Sqrt(Math.Abs(x)));
            }
            individual.Fitness = (float)fitnessValue;
        }
    }
}