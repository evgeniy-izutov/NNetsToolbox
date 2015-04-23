using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public interface ISetWeightsGenerator<T> where T:TrainPair {
		void GenerateWeights(List<T> set);
	}
}
