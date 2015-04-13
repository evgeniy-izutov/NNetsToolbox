using NeuralNet.LeanFactorStrategy;
using NeuralNet.RegularizationFunctions;
using StandardTypes;

namespace NeuralNet {
	public sealed class TrainProperties : ITrainProperties {
		public IMetrics Metrics { get; set; }
		public Regularization Regularization { get; set; }
		public float Epsilon { get; set; }
		public float CvLimit { get; set; }
		public int SkipCvLimitFirstIterations { get; set; }
		public float CvSlidingFactor { get; set; }
		public int MaxIterationCount { get; set; }
		public int PackageSize { get; set; }
		public float BaseLearnSpeed { get; set; }
		public float SpeedBonus { get; set; }
		public float SpeedPenalty { get; set; }
		public float SpeedLowBorder { get; set; }
		public float SpeedUpBorder { get; set; }
		public ILearnFactorStrategy LearnFactorStrategy { get; set; }
		public ILearnFactorStrategy AddedLearnFactorStrategy { get; set; }
		public float AverageLearnFactor { get; set; }
		public float Momentum { get; set; }
	}
}
