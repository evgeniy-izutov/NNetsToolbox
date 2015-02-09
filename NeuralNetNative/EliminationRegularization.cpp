#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "EliminationRegularization.h"

namespace NeuralNetNative {
	EliminationRegularization::EliminationRegularization(float regularizationFactor, float alpha) : Regularization(regularizationFactor) {
		_sqrAlpha = alpha*alpha;
	}

	float EliminationRegularization::GetDerivative(float value) {
		float tmp = _sqrAlpha + value*value;
		return 2.0f*RegularizationFactor*_sqrAlpha*value/(tmp*tmp);
	}
}