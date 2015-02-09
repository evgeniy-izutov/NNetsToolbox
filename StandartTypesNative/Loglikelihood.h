#pragma once

#include "Metrics.h"
#include "ExportDll.h"

namespace StandardTypesNative {
	class STANDARDTYPES_EXPORT LoglikelihoodForSoftmax : public Metrics {
	public:
		virtual float Calculate(const float* realOtput, const float* reconstructedOutput, int length) const;
		virtual void CalculatePartialDerivaitve(const float* realOtput, const float* reconstructedOutput, float* partialDerivaitve, int length) const;
	};
}