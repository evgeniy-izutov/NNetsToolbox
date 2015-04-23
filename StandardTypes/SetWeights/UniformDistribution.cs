using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public sealed class UniformDistribution<T> : ISetWeightsGenerator<T> where T:TrainPair {
		public void GenerateWeights(List<T> set) {
			var value = 1f/set.Count;
			for (var i = 0; i < set.Count; i++) {
				set[i].Weight = value;
			}
		}
	}
}
