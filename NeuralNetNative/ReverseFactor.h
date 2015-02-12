#pragma once

#include "ExportDll.h"
#include "LearnFactorStrategy.h"

namespace NeuralNetNative {
	class NEURALNETNATIVE_EXPORT ReverseFactor : public LearnFactorStrategy {
	public:
		virtual float GetFactor(int iterNumber) const;
	};
}