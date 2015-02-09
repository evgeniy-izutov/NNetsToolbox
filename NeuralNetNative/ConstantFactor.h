#pragma once

#include "ExportDll.h"
#include "LearnFactorStrategy.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT ConstantFactor : public LearnFactorStrategy {
	private:
		float _constantValue;
	public:
		ConstantFactor(float constantValue = 1.0f);
		virtual float GetFactor(int iterNumber);
	};
}