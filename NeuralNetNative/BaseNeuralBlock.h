#pragma once

#include "ExportDll.h"
#include "ActivationFunction.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT BaseNeuralBlock {
	protected:
		int Size;
		int PreviousSize;
		BaseNeuralBlock *Parent;
		ActivationFunction *Function;
        float *Weights;
        float *Bias;
        float *State;
        float *Net;
	protected: 
		BaseNeuralBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function);
        BaseNeuralBlock(int size, int parentSize, ActivationFunction *function);
	public:
		virtual ~BaseNeuralBlock(void);
        float* GetState(void);
        float* GetNet(void);
        BaseNeuralBlock* GetParent(void);
        float* GetWeights(void);
        void SetWeights(float *newWeights);
        float* GetBias(void);
        ActivationFunction* GetActivationFunction(void);
        int GetSize(void);
		int GetPreviousSize(void);
        virtual void Calculate(void) = 0;
        virtual void Calculate(const float *input) = 0;
	};
}