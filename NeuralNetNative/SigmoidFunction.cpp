#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "SigmoidFunction.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace tbb;

namespace NeuralNetNative {
	SigmoidFunction::SigmoidFunction(float alpha) {
		_alpha = alpha;
	}

	float SigmoidFunction::Calculate(float x) {
		return (1.0f/(1.0f + expf(-_alpha*x)));
	}

	float SigmoidFunction::CalculateFirstDerivative(float x) {
		float tmp = (1.0f/(1.0f + expf(-_alpha*x)));
		return _alpha*tmp*(1.0f - tmp);
	}

	float SigmoidFunction::CalculateFirstDerivative(const float *state, int index, int stateLength) {
		return _alpha*state[index]*(1.0f - state[index]);
	}

	void SigmoidFunction::CalculateFirstDerivative(float* target, const float* factors, const float* state, int stateLength) {
		parallel_for(blocked_range<size_t>(0, stateLength, SigmoidFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] = factors[i]*_alpha*state[i]*(1.0f - state[i]);
			}
		});
	}

	void SigmoidFunction::CalculateFirstDerivative(float* target, const float* state, int stateLength) {
		parallel_for(blocked_range<size_t>(0, stateLength, SigmoidFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] *= _alpha*state[i]*(1.0f - state[i]);
			}
		});
	}

	float SigmoidFunction::CalculateInvers(float y) {
		return (logf(y/(y - 1)))/_alpha;
	}
}