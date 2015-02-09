#define NEURALNETNATIVEAPI
#include "BaseNeuralBlock.h"
#include "malloc.h"

namespace NeuralNetNative {
	BaseNeuralBlock::BaseNeuralBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function) {
        Size = size;
		PreviousSize = parent->GetSize();
		Parent = parent;
		Function = function;

		Bias = (float*)_mm_malloc(size*sizeof(float), 32);
        State = (float*)_mm_malloc(size*sizeof(float), 32);
        Net = (float*)_mm_malloc(size*sizeof(float), 32);
		Weights = (float*)_mm_malloc(size*PreviousSize*sizeof(float), 32);
    }

    BaseNeuralBlock::BaseNeuralBlock(int size, int previousSize, ActivationFunction *function) {
        Size = size;
		PreviousSize = previousSize;
		Parent = 0;
		Function = function;

		Bias = (float*)_mm_malloc(size*sizeof(float), 32);
        State = (float*)_mm_malloc(size*sizeof(float), 32);
        Net = (float*)_mm_malloc(size*sizeof(float), 32);
		Weights = (float*)_mm_malloc(size*PreviousSize*sizeof(float), 32);
    }

	BaseNeuralBlock::~BaseNeuralBlock() {
		Size = 0;
		Parent = 0;
		Function = 0;

		_mm_free(Bias);
		_mm_free(State);
		_mm_free(Net);
		_mm_free(Weights);
	}

    float* BaseNeuralBlock::GetState(void) {
        return State;
    }

    float* BaseNeuralBlock::GetNet(void) {
        return Net;
    }

    BaseNeuralBlock* BaseNeuralBlock::GetParent(void) {
        return Parent;
    }

    float* BaseNeuralBlock::GetWeights(void) {
        return Weights;
    }

    void BaseNeuralBlock::SetWeights(float *newWeights) {
        if (Weights != 0) {
			_mm_free(Weights);
		}
		Weights = newWeights;
    }

    float* BaseNeuralBlock::GetBias(void) {
        return Bias;
    }

    ActivationFunction* BaseNeuralBlock::GetActivationFunction(void) {
        return Function;
    }

    int BaseNeuralBlock::GetSize(void) {
        return Size;
    }

	int BaseNeuralBlock::GetPreviousSize(void) {
        return PreviousSize;
    }
}