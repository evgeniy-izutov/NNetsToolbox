#pragma once

#include "TrainProperties.h"

using namespace NeuralNet;
using namespace StandardTypes;
using namespace System;
using namespace System::Collections::Generic;

namespace NeuralNetNativeWrapper {
    generic<class T> where T:TrainData
    public ref class TrainMethodNative abstract : public ITrainMethod<T> {
	internal:
		NeuralNetNative::TrainProperties *_nativeTrainProperties;
	protected:
		ITrainProperties<T>^ _properties;
	public:
		virtual void InitilazeMethod(INeuralNet^ neuralNet, ITrainProperties<T>^ trainProperties);
		virtual void Start(void) abstract;
		virtual void Stop(void) abstract;
		property ITrainProperties<T>^ Properties { virtual ITrainProperties<T>^ get(void); };
		virtual event EventHandler<IterationCompletedEventArgs^>^ IterationCompleted;
		virtual event EventHandler<IterativeProcessFinishedEventArgs^>^ IterativeProcessFinished;
	protected:
		virtual void CreateNativeNeuralNet(INeuralNet^ neuralNet) abstract;
		virtual void InitilazeNativeAlgorithm(void) abstract;
		virtual void DeleteNativeNeuralNet(void) abstract;
		void CreateNativeTrainProperties(ITrainProperties<T>^ trainProperties);
		void DeleteNativeProperties(void);
		void IterationCompletedHandler(int iterationNum, float iterationValue, float addedIterationValue);
		void IterativeProcessFinishedHandler(int iterationCount);
	};
}