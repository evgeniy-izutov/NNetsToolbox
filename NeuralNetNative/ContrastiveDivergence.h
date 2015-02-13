#pragma once

#include "ExportDll.h"
#include "RbmTrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainSingle.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT ContrastiveDivergence : public RbmTrainMethod {
		private:
			int _methodStepsCount;
			float *_oldDeltaWeights;
			float *_learnFactors;
			float *_derivativeAverages;
			float *_oldDeltaWeightsForVisibleBias;
			float *_learnFactorsForVisibleBias;
			float *_oldDeltaWeightsForHiddenBias;
			float *_learnFactorsForHiddenBias;
			float *_derivativeAveragesForVisibleBias;
			float *_derivativeAveragesForHiddenBias;
		public:
			ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                  int trainDataSize,
                                  GradientFunction *gradientFunction,
                                  int methodStepsCount);
			ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                  StandardTypesNative::TrainSingle **testData,
                                  int trainDataSize, int testDataSize,
                                  GradientFunction *gradientFunction,
                                  int methodStepsCount);
			virtual ~ContrastiveDivergence(void);
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