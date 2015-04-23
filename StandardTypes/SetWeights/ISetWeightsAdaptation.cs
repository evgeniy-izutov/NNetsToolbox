using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public interface ISetWeightsAdaptation<T> where T:TrainData {
		void ChangeWeights(IList<T> set, float[] errors);
	}
}
