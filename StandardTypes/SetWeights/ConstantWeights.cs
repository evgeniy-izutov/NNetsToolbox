using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public sealed class ConstantWeights<T> : ISetWeightsAdaptation<T> where T:TrainData {
		public void ChangeWeights(List<T> set, float[] errors) {
		}
	}
}
