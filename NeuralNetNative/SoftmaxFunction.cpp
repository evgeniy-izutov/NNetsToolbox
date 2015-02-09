#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "SoftmaxFunction.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace tbb;

namespace NeuralNetNative {
	SoftmaxFunction::SoftmaxFunction(void) {
	}

	float SoftmaxFunction::Calculate(float x) {
		return 0.0f;
	}

	float SoftmaxFunction::CalculateFirstDerivative(float x) {
		return 0.0f;
	}

	float SoftmaxFunction::CalculateFirstDerivative(const float *state, int index, int stateLength) {
		//return state[index]*(1.0f - state[index]);
		return 0.0f;
	}

	void SoftmaxFunction::CalculateFirstDerivative(float* target, const float* factors, const float* state, int stateLength) {
		/*parallel_for(blocked_range<size_t>(0, stateLength, SoftmaxFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] = factors[i]*state[i]*(1.0f - state[i]);
			}
		});*/
		for (int i = 0; i < stateLength; i++) {
			target[i] = factors[i];
		}
	}

	void SoftmaxFunction::CalculateFirstDerivative(float* target, const float* state, int stateLength) {
		/*parallel_for(blocked_range<size_t>(0, stateLength, SoftmaxFunctionGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int i = r.begin(); i < r.end(); i++) {
				target[i] *= state[i]*(1.0f - state[i]);
			}
		});*/
	}

	float SoftmaxFunction::CalculateInvers(float y) {
		return 0.0f;
	}
}