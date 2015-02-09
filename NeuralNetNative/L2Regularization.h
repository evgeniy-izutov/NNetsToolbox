#pragma once

#include "ExportDll.h"
#include "Regularization.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT L2Regularization : public Regularization {
	public:
		L2Regularization(float regularizationFactor);
		virtual float GetDerivative(float value);
	};
}