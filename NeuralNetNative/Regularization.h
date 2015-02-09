#pragma once

#include "ExportDll.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT Regularization {
	public:
		virtual float GetDerivative(float value) = 0;
	protected:
		float RegularizationFactor;
		Regularization(float regularizationFactor);
	};
}