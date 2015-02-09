#define NEURALNETNATIVEAPI
#include "L2Regularization.h"

namespace NeuralNetNative {
	L2Regularization::L2Regularization(float regularizationFactor) : Regularization(regularizationFactor) {
	}

	float L2Regularization::GetDerivative(float value) {
		return RegularizationFactor*value;
	}
}