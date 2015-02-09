#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "SqrtReverseFactor.h"

namespace NeuralNetNative {
	float SqrtReverseFactor::GetFactor(int iterNumber) {
		const float Epsilon = 0.000001f;
		return invsqrtf(iterNumber + Epsilon);
	}
}