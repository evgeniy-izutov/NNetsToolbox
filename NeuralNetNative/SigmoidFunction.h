#pragma once

#include "ExportDll.h"
#include "ActivationFunction.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT SigmoidFunction : public ActivationFunction {
	private:
		float _alpha;
	public:
		SigmoidFunction(float alpha);
		virtual float Calculate(float x);
		virtual float CalculateFirstDerivative(float x);
		virtual float CalculateFirstDerivative(const float *state, int index, int stateLength);
		virtual void CalculateFirstDerivative(float* target, const float* factors, const float* state, int stateLength);
	    virtual void CalculateFirstDerivative(float* target, const float* state, int stateLength);
		virtual float CalculateInvers(float y);
	};
}