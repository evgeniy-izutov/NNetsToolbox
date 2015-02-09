#pragma once

#include "ExportDll.h"
#include "BaseNeuralBlock.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT SoftmaxSimpleNeuronBlock : public BaseNeuralBlock {
	public:
		SoftmaxSimpleNeuronBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function);
		SoftmaxSimpleNeuronBlock(int size, int parentSize, ActivationFunction *function);
		virtual void Calculate(void);
		virtual void Calculate(const float *input);
	};
}