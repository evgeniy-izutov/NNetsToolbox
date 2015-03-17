#pragma once

#include "ExportDll.h"
#include "TrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainSingle.h"
#include "RestrictedBoltzmannMachine.h"
#include "RbmGradients.h"
#include "GradientFunction.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
	    class NEURALNETNATIVE_EXPORT RbmTrainMethod : public TrainMethod {
		private:
		    StandardTypesNative::RandomAccessIterator<StandardTypesNative::TrainSingle*> *_trainDataIterator;
			StandardTypesNative::TrainSingle **_testData;
			int _testDataSize;
			float _packageFactor;
			float *_neuronNetOutput;
            GradientFunction *_gradientFunction;
		protected:
		    TrainProperties *properties;
            RbmGradients *gradients;
			RestrictedBoltzmannMachineBase *neuralNet;
			int visibleStatesCount;
			int hiddenStatesCount;
			int epochNumber;
			int packagesCount;
        protected:
        	RbmTrainMethod(StandardTypesNative::TrainSingle **trainData,
                           int trainDataSize,
                           GradientFunction *gradientFunction);
            RbmTrainMethod(StandardTypesNative::TrainSingle **trainData,
                           StandardTypesNative::TrainSingle **testData,
                           int trainDataSize, int testDataSize,
                           GradientFunction *gradientFunction);
        	virtual ~RbmTrainMethod(void);
            virtual void RunIterativeProcess(void);
			virtual void ApplyResults(void);
            virtual void CreateTemporaryData(void) = 0;
			virtual void DeleteTemporaryData(void) = 0;
            virtual void MakePositivePhase(float *input) = 0;
		    virtual void MakeNegativePhase(int packageId) = 0;
		    virtual float* GetVisibleStatesOnNegativePhase(int packageId) = 0;
		    virtual float* GetHiddenStatesOnNegativePhase(void) = 0;
		    virtual void RestoreVisibleStates(int packageId) = 0;
            virtual void ModifyWeightsOfNeuronNet() = 0;
        private:
            bool IsTestDataAvailable() const;
            void RunTraingWithTesting(void);
            void RunTraingWithoutTesting(void);
            int CalculatePackagesCount(void) const;
            float TestModel(StandardTypesNative::TrainSingle **data, int dataSize) const;
            void TrainEpoch(void);
			void TrainPackage(int packageId);
        public:
            void InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties);
			TrainProperties* Properties(void) const;
        };
	}
}
