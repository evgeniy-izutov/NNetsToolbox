#define STANDARDTYPESAPI
#include <mathimf.h>
#include "HammingDistance.h"
#include <cfloat>

namespace StandardTypesNative {
	float HammingDistance::Calculate(const float* realOtput, const float* reconstructedOutput, int length) const {
		int num = 0;
		for (int i = 0; i < length; i++) {
			if (fabsf(realOtput[i] - reconstructedOutput[i]) > FLT_EPSILON) {
				num++;
			}
		}
		return num;
	}

	void HammingDistance::CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const {
		#pragma simd
		for (int i = 0; i < length; i++) {
			partialDerivaitve[i] = 0.0f;
		}
	}
}