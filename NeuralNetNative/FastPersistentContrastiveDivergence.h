#pragma once

#include "ExportDll.h"
#include "TrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainSingle.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		class NEURALNETNATIVE_EXPORT FastPersistentContrastiveDivergence : public TrainMethod {
		private:
			StandardTypesNative::RandomAccessIterator<StandardTypesNative::TrainSingle*> *_trainDataIterator;
			StandardTypesNative::TrainSingle **_testData;
			int _testDataSize;
			TrainProperties *_properties;
			RestrictedBoltzmannMachineBase *_neuralNet;
			float _fastWeightsDecreaseFactor;
			float _packageFactor;
			int _visibleStatesCount;
			int _hiddenStatesCount;
			float *_persistentVisibleStates;
			float *_fastWeights;
			float *_fastWeightsForVisibleBias;
			float *_fastWeightsForHiddenBias;
			float *_oldDeltaRegularWeights;
			float *_oldDeltaRegularWeightsForVisibleBias;
			float *_oldDeltaRegularWeightsForHiddenBias;
			float *_packageDerivative;
			float *_packageDerivativeForVisibleBias;
			float *_packageDerivativeForHiddenBias;
			float *_neuronNetOutput;
			int _epochNumber;
			int _packagesCount;
		public:
			FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, int trainDataSize, float fastWeightsDecreaseFactor);
			FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, StandardTypesNative::TrainSingle **testData, int trainDataSize, int testDataSize, float fastWeightsDecreaseFactor);
			~FastPersistentContrastiveDivergence(void);
			void InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties);
			TrainProperties* Properties(void);
		protected:
			void RunIterativeProcess(void);
			void ApplyResults(void);
		private:
			void AllocateMemory(void);
			void ClearData(void);
			int CalculatePackagesCount(void);
			void TrainEpoch(void);
			void TrainPackage(int packageId);
			void MakePositivePhase(float *input);
			void MakeNegativePhase(int packageId);
			void ModifyWeightsOfNeuronNet(void);
			float TestModel(StandardTypesNative::TrainSingle **data, int dataSize);
		};
	}
}