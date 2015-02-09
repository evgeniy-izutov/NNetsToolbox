#pragma once

#include "ExportDll.h"
#include "Regularization.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT EliminationRegularization : public Regularization {
	private:
		float _sqrAlpha;
	public:
		EliminationRegularization(float regularizationFactor, float alpha);
		virtual float GetDerivative(float value);
	};
}