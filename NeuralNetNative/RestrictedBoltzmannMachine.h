#pragma once

#include "ExportDll.h"
#include "NeuralNet.h"
#include <random>

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT RestrictedBoltzmannMachineBase : public NeuralNet {
		protected:
			std::mt19937 *_randomDevice;
			std::uniform_real_distribution<float> *_uniformDistribution;
			int _visibleStatesCount;
			int _hiddenStatesCount;
			float *_visibleStates;
			float *_hiddenStates;
			float *_weights;
			float *_visibleStatesBias;
			float *_hiddenStatesBias;
		public:
			RestrictedBoltzmannMachineBase(int visibleStatesCount, int hiddenStatesCount);
			~RestrictedBoltzmannMachineBase(void);
			virtual void VisibleLayerCalculateActivity(void) = 0;
			virtual void HiddenLayerCalculateActivity(void) = 0;
			virtual void HiddenLayerCalculateActivity(const float *newVisibleState) = 0;
			virtual void VisibleLayerCalculateActivity(const float *addedWeight, const float *addedVisibleBias) = 0;
			virtual void HiddenLayerCalculateActivity(const float *addedWeight, const float *addedHiddenBias) = 0;
			virtual void HiddenLayerCalculateActivity(const float *newVisibleState, const float *addedWeight, const float *addedHiddenBias) = 0;
			virtual void VisibleLayerSampling(void);
			virtual void HiddenLayerSampling(void);
			virtual void VisibleLayerSampling(float *target);
			virtual void HiddenLayerSampling(float *target);
			void VisibleLayerCopyTo(float *target);
			void HiddenLayerCopyTo(float *target);
			void Predict(const float *input, float *output);
			int GetVisibleStatesCount(void);
			int GetHiddenStatesCount(void);
			float* GetWeights(void);
			float* GetVisibleStatesBias(void);
			float* GetHiddenStatesBias(void);
			float* GetVisibleStates(void);
			float* GetHiddenStates(void);
		private:
			void SetOutput(float *output);
		};
	}
}