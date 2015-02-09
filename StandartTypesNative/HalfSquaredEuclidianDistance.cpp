#define STANDARDTYPESAPI
#include "HalfSquaredEuclidianDistance.h"

namespace StandardTypesNative {
	float HalfSquaredEuclidianDistance::Calculate(const float* realOtput, const float* reconstructedOutput, int length) const {
		float dif;
		float value = 0.0f;

		for (int i = 0; i < length; i++) {
			dif = realOtput[i] - reconstructedOutput[i];
			value += dif*dif;
		}
		return 0.5f*value;
	}

	void HalfSquaredEuclidianDistance::CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const {
		#pragma simd
		for (int i = 0; i < length; i++) {
			partialDerivaitve[i] = reconstructedOutput[i] - realOtput[i];
		}
	}
}