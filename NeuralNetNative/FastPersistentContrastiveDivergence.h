#pragma once

#include "ExportDll.h"
#include "RbmTrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainSingle.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT FastPersistentContrastiveDivergence : public RbmTrainMethod {
		private:
			float _fastWeightsDecreaseFactor;
			float *_persistentVisibleStates;
			float *_fastWeights;
			float *_fastWeightsForVisibleBias;
			float *_fastWeightsForHiddenBias;
			float *_oldDeltaRegularWeights;
			float *_oldDeltaRegularWeightsForVisibleBias;
			float *_oldDeltaRegularWeightsForHiddenBias;
		public:
			FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                                int trainDataSize,
                                                GradientFunction *gradientFunction,
                                                float fastWeightsDecreaseFactor);
			FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                                StandardTypesNative::TrainSingle **testData,
                                                int trainDataSize, int testDataSize,
                                                GradientFunction *gradientFunction,
                                                float fastWeightsDecreaseFactor);
			virtual ~FastPersistentContrastiveDivergence(void);
        protected:
            virtual void CreateTemporaryData(void);
			virtual void DeleteTemporaryData(void);
            virtual void MakePositivePhase(float *input);
		    virtual void MakeNegativePhase(int packageId);
		    virtual float* GetVisibleStatesOnNegativePhase(int packageId);
		    virtual float* GetHiddenStatesOnNegativePhase(void);
		    virtual void RestoreVisibleStates(int packageId);
            virtual void ModifyWeightsOfNeuronNet();
		};
	}
}