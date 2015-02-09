#define NEURALNETNATIVEAPI
#include "ConstantFactor.h"

namespace NeuralNetNative {
	ConstantFactor::ConstantFactor(float constantValue) {
		_constantValue = constantValue;
	}
	
	float ConstantFactor::GetFactor(int iterNumber) {
		return _constantValue;
	}
}