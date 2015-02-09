#pragma once

#include "NeuralNet.h"
#include "TrainProperties.h"
#include "ItarativeProcess.h"

namespace NeuralNetNative {
	class TrainMethod : public StandardTypesNative::ItarativeProcess {
	public:
		virtual void InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties) = 0;
		virtual TrainProperties* Properties(void) = 0;
	};
}