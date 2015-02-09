#define NEURALNETNATIVEAPI
#include "NoRegularization.h"

namespace NeuralNetNative {
	NoRegularization::NoRegularization(void) : Regularization(0.0f) {
	}

	float NoRegularization::GetDerivative(float value) {
		return 0.0f;
	}
}