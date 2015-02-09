#pragma once

#include "NeuralNet.h"

namespace NeuralNetNative {
	class NeuralNetFactory {
		virtual NeuralNet* CreateNeuralNet(void) = 0;
	};

	enum StartWeightGenerator {
    	NullDistribution,
		UniformDistribution,
		NormalDistribution
    };
}