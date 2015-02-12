#pragma once

#include "ExportDll.h"
#include "LearnFactorStrategy.h"

namespace NeuralNetNative {
    class NEURALNETNATIVE_EXPORT LinearFactor : public LearnFactorStrategy {
    public:
    	LinearFactor(float startFactor, float endFactor, int stepsCount);
		LinearFactor(float a, float b);
    	virtual float GetFactor(int iterNumber) const;
	private:
	    float _a;
		float _b;
    };
}
