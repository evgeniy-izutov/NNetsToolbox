using System.Collections.Generic;

namespace StandardTypes {
	public interface IDataLoader {
		List<TrainSingle> LoadData(List<FeatureDescription> inputDescription);
		List<TrainPair> LoadData(List<FeatureDescription> inputDescription, List<FeatureDescription> outputDescription);
	}
}