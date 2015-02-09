#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "L1Regularization.h"

namespace NeuralNetNative {
	L1Regularization::L1Regularization(float regularizationFactor) : Regularization(regularizationFactor) {
	}

	float L1Regularization::GetDerivative(float value) {
		return copysignf(RegularizationFactor, value);
	}
}