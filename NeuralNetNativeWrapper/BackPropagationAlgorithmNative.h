#pragma once

#include "TrainMethodNative.h"
#include "BackPropagationAlgorithm.h"

using namespace NeuralNet;
using namespace StandardTypes;
using namespace System;
using namespace System::Collections::Generic;

namespace NeuralNetNativeWrapper {
	namespace MultyLayerPerceptronNativeWrapper {
        public ref class BackPropagationAlgorithmNative	: public TrainMethodNative<TrainPair^>, public System::IDisposable {
		internal:
			NeuralNetNative::MultyLayerPerceptron::BackPropagationAlgorithm *_nativeAlgorithm;
			NeuralNetNative::MultyLayerPerceptron::MultyLayerPerceptron *_nativeNeuralNet;
			StandardTypesNative::TrainPair **_nativeTrainData;
			StandardTypesNative::TrainPair **_nativeTestData;
			int _nativeTrainDataSize;
			int _nativeTestDataSize;
			NeuralNetNative::ActivationFunction *_hiddenActivationFunction;
			NeuralNetNative::ActivationFunction *_outputActivationFunction;
			int *_nativeLayersStruct;
		protected:
			MultyLayerPerceptron::MultyLayerPerceptron^ _multyLayerPerceptron;
		public:
			BackPropagationAlgorithmNative(IList<TrainPair^>^ trainData);
			BackPropagationAlgorithmNative(IList<TrainPair^>^ trainData, IList<TrainPair^>^ testData);
			~BackPropagationAlgorithmNative(void);
			virtual void Start(void) override;
			virtual void Stop(void) override;
		protected:
			virtual void CreateNativeNeuralNet(INeuralNet^ neuralNet) override;
			virtual void DeleteNativeNeuralNet(void) override;
			virtual void InitilazeNativeAlgorithm(void) override;
			void ApplyResult(void);
			void DeleteNativeAlgorithm(void);
			void AllocateNativeTrainData(IList<TrainPair^>^ trainData);
			void AllocateNativeTestData(IList<TrainPair^>^ testData);
			void DeleteNativeTrainData(void);
			void DeleteNativeTestData(void);
		};
	}
}