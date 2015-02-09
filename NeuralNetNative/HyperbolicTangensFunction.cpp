#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "HyperbolicTangensFunction.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace tbb;

namespace NeuralNetNative {
	HyperbolicTangensFunction::HyperbolicTangensFunction(float alpha, float betta) {
		_alpha = alpha;
		_betta = betta;
		_derivativeFactor = _betta/_alpha;
	}

	float HyperbolicTangensFunction::Calculate(float x) {
		return (_alpha*tanhf(_betta*x));
	}

	float HyperbolicTangensFunction::CalculateFirstDerivative(float x) {
		float tmp = tanhf(_betta*x);
		return _alpha*_betta*(1.0f - tmp*tmp);
	}

	float HyperbolicTangensFunction::CalculateFirstDerivative(const float *state, int index, int stateLength) {
		return _derivativeFactor*(_alpha - state[index])*(_alpha + state[index]);
	}

	void HyperbolicTangensFunction::CalculateFirstDerivative(float* target, const float* factors, const float* state, int stateLength) {
		parallel_for(blocked_range<size_t>(0, stateLength, HyperbolicTangensFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] = factors[i]*_derivativeFactor*(_alpha - state[i])*(_alpha + state[i]);
			}
		});
	}

	void HyperbolicTangensFunction::CalculateFirstDerivative(float* target, const float* state, int stateLength) {
		parallel_for(blocked_range<size_t>(0, stateLength, HyperbolicTangensFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] *= _derivativeFactor*(_alpha - state[i])*(_alpha + state[i]);
			}
		});
	}

	float HyperbolicTangensFunction::CalculateInvers(float y) {
		return atanhf(y/_alpha)/_betta;
	}
}