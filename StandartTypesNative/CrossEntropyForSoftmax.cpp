#define STANDARDTYPESAPI
#include <mathimf.h>
#include "CrossEntropy.h"

namespace StandardTypesNative {
	float CrossEntropyForSoftmax::Calculate(const float* realOtput, const float* reconstructedOutput, int length) const {
		float d = 0.0f;
		for (int i = 0; i < length; i++) {
			d += realOtput[i]*logf(reconstructedOutput[i]);
		}
		return -d;
	}

	void CrossEntropyForSoftmax::CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const {
		#pragma simd
		for (int i = 0; i < length; i++) {
			partialDerivaitve[i] = reconstructedOutput[i] - realOtput[i];
		}
	}
}