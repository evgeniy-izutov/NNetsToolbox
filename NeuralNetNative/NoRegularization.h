#pragma once

#include "ExportDll.h"
#include "Regularization.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT NoRegularization : public Regularization {
	public:
		NoRegularization(void);
		virtual float GetDerivative(float value);
	};
}