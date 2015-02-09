#pragma once

namespace StandardTypesNative {
	class Metrics {
	public:
		virtual float Calculate(const float* realOtput, const float* reconstructedOutput, int length) const = 0;
		virtual void CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const = 0;
	};
}