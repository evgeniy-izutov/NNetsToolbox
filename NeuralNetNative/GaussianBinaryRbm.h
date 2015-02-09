#pragma once

#include "ExportDll.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT GaussianBinaryRbm : public RestrictedBoltzmannMachineBase {
		private:
			std::normal_distribution<float> *_normalDistribution;
		public:
			GaussianBinaryRbm(int visibleStatesCount, int hiddenStatesCount);
			virtual ~GaussianBinaryRbm(void);
			virtual void VisibleLayerCalculateActivity(void);
			virtual void HiddenLayerCalculateActivity(void);
			virtual void HiddenLayerCalculateActivity(const float *newVisibleState);
			virtual void VisibleLayerCalculateActivity(const float *addedWeight, const float *addedVisibleBias);
			virtual void HiddenLayerCalculateActivity(const float *addedWeight, const float *addedHiddenBias);
			virtual void HiddenLayerCalculateActivity(const float *newVisibleState, const float *addedWeight, const float *addedHiddenBias);
			virtual void VisibleLayerSampling(void);
			virtual void VisibleLayerSampling(float *target);
		};
	}
}