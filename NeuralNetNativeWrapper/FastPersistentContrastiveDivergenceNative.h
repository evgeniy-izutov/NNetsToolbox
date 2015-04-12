#pragma once

#include "TrainMethodNative.h"
#include "RestrictedBoltzmannMachine.h"
#include "FastPersistentContrastiveDivergence.h"

using namespace NeuralNet;
using namespace StandardTypes;
using namespace System;
using namespace System::Collections::Generic;
using namespace NeuralNet::GenerativeRbm;

namespace NeuralNetNativeWrapper {
	namespace RestrictedBoltzmannMachineNativeWrapper {
		public ref class FastPersistentContrastiveDivergenceNative : public TrainMethodNative, public System::IDisposable {
		internal:
			NeuralNetNative::RestrictedBoltzmannMachine::FastPersistentContrastiveDivergence *_nativeAlgorithm;
			NeuralNetNative::RestrictedBoltzmannMachine::RestrictedBoltzmannMachineBase *_nativeNeuralNet;
            NeuralNetNative::RestrictedBoltzmannMachine::GradientFunction *_nativeGradientFunction;
			StandardTypesNative::TrainSingle **_nativeTrainData;
			StandardTypesNative::TrainSingle **_nativeTestData;
			int _nativeTrainDataSize;
			int _nativeTestDataSize;
		protected:
			RestrictedBoltzmannMachine^ _restrictedBoltzmannMachine;
		public:
			FastPersistentContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                                      IGradientFunction^ gradient,
                                                      float fastWeightsDecreaseFactor);
			FastPersistentContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                                      IList<TrainSingle^>^ testData,
                                                      IGradientFunction^ gradient,
                                                      float fastWeightsDecreaseFactor);
			~FastPersistentContrastiveDivergenceNative(void);
			virtual void Start(void) override;
			virtual void Stop(void) override;
		protected:
			virtual void CreateNativeNeuralNet(INeuralNet^ neuralNet) override;
			virtual void DeleteNativeNeuralNet(void) override;
			virtual void InitilazeNativeAlgorithm(void) override;
			void ApplyResult(void);
			void DeleteNativeAlgorithm(void);
			void AllocateNativeTrainData(IList<TrainSingle^>^ trainData);
			void AllocateNativeTestData(IList<TrainSingle^>^ testData);
			void DeleteNativeTrainData(void);
            void DeleteNativeTestData(void);
            void AllocateNativeGradientFunction(IGradientFunction^ gradient);
            void DeleteNativeGradientFunction(void);
		};
	}
}
