#define NEURALNETNATIVEAPI
#include "LinearFactor.h"

namespace NeuralNetNative {
	LinearFactor::LinearFactor(float startFactor, float endFactor, int stepsCount) {
	    if (stepsCount <= 1) {
            _a = startFactor;
            _b = 0;
        }
        else {
            _a = (endFactor - startFactor)/(stepsCount - 1);
            _b = startFactor - _a;
        }
	}

	LinearFactor::LinearFactor(float a, float b) {
	    _a = a;
		_b = b;
	}
	
	float LinearFactor::GetFactor(int iterNumber) const {
		return _a*iterNumber + _b;
	}
}
