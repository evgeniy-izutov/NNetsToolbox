using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public sealed class ConstantWeights<T> : ISetWeightsAdaptation<T> where T:TrainData {
		public void ChangeWeights(IList<T> set, float[] errors) {
		}
	}
}
