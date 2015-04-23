using NeuralNet.RegularizationFunctions;
using StandardTypes;
using StandardTypes.FactorStrategy;
using StandardTypes.SetWeights;

namespace NeuralNet {
	public sealed class TrainProperties<T> : ITrainProperties<T> where T:TrainData {
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
		public IFactorStrategy LearnFactorStrategy { get; set; }
		public IFactorStrategy AddedLearnFactorStrategy { get; set; }
		public float AverageLearnFactor { get; set; }
		public float Momentum { get; set; }
		public ISetWeightsAdaptation<T> SetWeightsAdaptation { get; set; }

		public TrainProperties() {
			Metrics = new SquaredEuclidianDistance();
			Regularization = new NoRegularization();
			LearnFactorStrategy = new ConstantFactor();
			AddedLearnFactorStrategy = new ConstantFactor();
			SetWeightsAdaptation = new ConstantWeights<T>();
		}
	}
}
