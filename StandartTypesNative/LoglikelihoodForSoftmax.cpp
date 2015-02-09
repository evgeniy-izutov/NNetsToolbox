#define STANDARDTYPESAPI
#include <mathimf.h>
#include "Loglikelihood.h"

namespace StandardTypesNative {
	float LoglikelihoodForSoftmax::Calculate(const float* realOtput, const float* reconstructedOutput, int length) const {
		float d = 0.0f;
		#pragma simd
		for (int i = 0; i < length; i++) {
			d += realOtput[i]*logf(reconstructedOutput[i]) + (1.0f - realOtput[i])*logf(1.0f - reconstructedOutput[i]);
		}
		return -d;
	}

	void LoglikelihoodForSoftmax::CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const {
		#pragma simd
		for (int i = 0; i < length; i++) {
			partialDerivaitve[i] = -realOtput[i]/reconstructedOutput[i] + (1.0f - realOtput[i])/(1.0f - reconstructedOutput[i]);
		}
	}
}