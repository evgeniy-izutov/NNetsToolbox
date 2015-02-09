#pragma once

#include "ExportDll.h"
#include "Regularization.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT L1Regularization : public Regularization {
	public:
		L1Regularization(float regularizationFactor);
		virtual float GetDerivative(float value);
	};
}