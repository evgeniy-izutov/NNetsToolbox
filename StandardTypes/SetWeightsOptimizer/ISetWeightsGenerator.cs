using System.Collections.Generic;

namespace StandardTypes.SetWeightsOptimizer {
	public interface ISetWeightsGenerator<T> where T:TrainPair {
		void GenerateWeights(List<T> set);
	}
}
