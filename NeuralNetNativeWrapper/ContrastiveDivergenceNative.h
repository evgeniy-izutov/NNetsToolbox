#pragma once

#include "TrainMethodNative.h"
#include "RestrictedBoltzmannMachine.h"
#include "ContrastiveDivergence.h"

using namespace NeuralNet;
using namespace StandardTypes;
using namespace System;
using namespace System::Collections::Generic;
using namespace NeuralNet::GenerativeRbm;

namespace NeuralNetNativeWrapper {
	namespace RestrictedBoltzmannMachineNativeWrapper {
        public ref class ContrastiveDivergenceNative : public TrainMethodNative<TrainSingle^>, public System::IDisposable {
		internal:
			NeuralNetNative::RestrictedBoltzmannMachine::ContrastiveDivergence *_nativeAlgorithm;
			NeuralNetNative::RestrictedBoltzmannMachine::RestrictedBoltzmannMachineBase *_nativeNeuralNet;
            NeuralNetNative::RestrictedBoltzmannMachine::GradientFunction *_nativeGradientFunction;
			StandardTypesNative::TrainSingle **_nativeTrainData;
			StandardTypesNative::TrainSingle **_nativeTestData;
			int _nativeTrainDataSize;
			int _nativeTestDataSize;
		protected:
			RestrictedBoltzmannMachine^ _restrictedBoltzmannMachine;
		public:
			ContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                        IGradientFunction^ gradient,
                                        int methodStepsCount);
			ContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                        IList<TrainSingle^>^ testData,
                                        IGradientFunction^ gradient,
                                        int methodStepsCount);
			~ContrastiveDivergenceNative(void);
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
