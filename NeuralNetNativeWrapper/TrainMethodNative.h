#pragma once

#include "TrainProperties.h"

using namespace NeuralNet;
using namespace StandardTypes;
using namespace System;
using namespace System::Collections::Generic;

namespace NeuralNetNativeWrapper {
	public ref class TrainMethodNative abstract : public ITrainMethod {
	internal:
		NeuralNetNative::TrainProperties *_nativeTrainProperties;
	protected:
		ITrainProperties^ _properties;
	public:
		virtual void InitilazeMethod(INeuralNet^ neuralNet, ITrainProperties^ trainProperties);
		virtual void Start(void) abstract;
		virtual void Stop(void) abstract;
		property ITrainProperties^ Properties { virtual ITrainProperties^ get(void); };
		virtual event EventHandler<IterationCompletedEventArgs^>^ IterationCompleted;
		virtual event EventHandler<IterativeProcessFinishedEventArgs^>^ IterativeProcessFinished;
	protected:
		virtual void CreateNativeNeuralNet(INeuralNet^ neuralNet) abstract;
		virtual void InitilazeNativeAlgorithm(void) abstract;
		virtual void DeleteNativeNeuralNet(void) abstract;
		void CreateNativeTrainProperties(ITrainProperties^ trainProperties);
		void DeleteNativeProperties(void);
		void IterationCompletedHandler(int iterationNum, float iterationValue, float addedIterationValue);
		void IterativeProcessFinishedHandler(int iterationCount);
	};
}