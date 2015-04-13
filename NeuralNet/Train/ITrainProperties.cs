using NeuralNet.LeanFactorStrategy;
using NeuralNet.RegularizationFunctions;
using StandardTypes;

namespace NeuralNet {
	public interface ITrainProperties {
		IMetrics Metrics { get; set; }
		Regularization Regularization { get; set; }
		float Epsilon { get; set; }
		float CvLimit { get; set; }
		int SkipCvLimitFirstIterations { get; set; }
		float CvSlidingFactor { get; set; }
		int MaxIterationCount { get; set; }
		int PackageSize { get; set; }
		float BaseLearnSpeed { get; set; }
		float SpeedBonus { get; set; }
		float SpeedPenalty { get; set; }
		float SpeedLowBorder { get; set; }
		float SpeedUpBorder { get; set; }
		ILearnFactorStrategy LearnFactorStrategy { get; set; }
		ILearnFactorStrategy AddedLearnFactorStrategy { get; set; }
		float AverageLearnFactor { get; set; }
		float Momentum { get; set; }
	}
}
