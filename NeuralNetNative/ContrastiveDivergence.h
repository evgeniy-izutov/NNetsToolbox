#pragma once

#include "ExportDll.h"
#include "TrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainSingle.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT ContrastiveDivergence : public TrainMethod {
		private:
			int _methodStepsCount;
			StandardTypesNative::RandomAccessIterator<StandardTypesNative::TrainSingle*> *_trainDataIterator;
			StandardTypesNative::TrainSingle **_testData;
			int _testDataSize;
			TrainProperties *_properties;
			RestrictedBoltzmannMachineBase *_neuralNet;
			float _packageFactor;
			int _visibleStatesCount;
			int _hiddenStatesCount;
			float *_oldDeltaWeights;
			float *_learnFactors;
			float *_derivativeAverages;
			float *_packageDerivative;
			float *_oldDeltaWeightsForVisibleBias;
			float *_learnFactorsForVisibleBias;
			float *_oldDeltaWeightsForHiddenBias;
			float *_learnFactorsForHiddenBias;
			float *_derivativeAveragesForVisibleBias;
			float *_packageDerivativeForVisibleBias;
			float *_derivativeAveragesForHiddenBias;
			float *_packageDerivativeForHiddenBias;
			float *_neuronNetOutput;
			int _epochNumber;
		public:
			ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, int trainDataSize, int methodStepsCount);
			ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, StandardTypesNative::TrainSingle **testData, int trainDataSize, int testDataSize, int methodStepsCount);
			~ContrastiveDivergence(void);
			void InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties);
			TrainProperties* Properties(void);
		protected:
			void RunIterativeProcess(void);
			void ApplyResults(void);
		private:
			void AllocateMemory(void);
			void ClearData(void);
			int CalculatePackagesCount(void);
			void TrainEpoch(int packagesCount);
			void TrainPackage(float regularizationFactorPerPackage);
			void CollectWeightsDelta(float *input);
			void MakePositivePhase(float *input);
			void MakeSampling(void);
			void MakeNegativePhase(void);
			void ModifyWeightsOfNeuronNet(float regularizationFactorPerPackage);
			float TestModel(StandardTypesNative::TrainSingle **data, int dataSize);
		};
	}
}