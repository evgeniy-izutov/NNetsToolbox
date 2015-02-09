#pragma once

#include "InvertibleFunction.h"

namespace NeuralNetNative {
	class ActivationFunction : public StandardTypesNative::InvertibleFunction {
	public:
		virtual float CalculateFirstDerivative(float x) = 0;
		virtual float CalculateFirstDerivative(const float *state, int index, int stateLength) = 0;
		virtual void CalculateFirstDerivative(float* target, const float* factors, const float* state, int stateLength) = 0;
	    virtual void CalculateFirstDerivative(float* target, const float* state, int stateLength) = 0;
	};
}