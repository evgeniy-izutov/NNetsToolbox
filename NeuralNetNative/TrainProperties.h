#pragma once

#include "Metrics.h"
#include "Regularization.h"
#include "LearnFactorStrategy.h"

namespace NeuralNetNative {
	struct TrainProperties {
	public:
		StandardTypesNative::Metrics *Metrics;
		Regularization *Regularization;
		float Epsilon;
		int MaxIterationCount;
		int PackageSize;
		float CvLimit;
		float BaseLearnSpeed;
		float SpeedBonus;
		float SpeedPenalty;
		float SpeedLowBorder;
		float SpeedUpBorder;
		LearnFactorStrategy *FactorStrategy;
		LearnFactorStrategy *AddedFactorStrategy;
		float AverageLearnFactor;
		float Momentum;
	};
}