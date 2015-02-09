#define NEURALNETNATIVEAPI
#include "ReverseFactor.h"

namespace NeuralNetNative {
	float ReverseFactor::GetFactor(int iterNumber) {
		const float Epsilon = 0.000001f;
		return 1.0f/(iterNumber + Epsilon);
	}
}