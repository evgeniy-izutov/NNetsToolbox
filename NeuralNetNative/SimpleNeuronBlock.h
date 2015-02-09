#pragma once

#include "ExportDll.h"
#include "BaseNeuralBlock.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT SimpleNeuronBlock : public BaseNeuralBlock {
	public:
		SimpleNeuronBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function);
		SimpleNeuronBlock(int size, int parentSize, ActivationFunction *function);
		void Calculate(void);
		virtual void Calculate(const float *input);
	};
}