#define NEURALNETNATIVEAPI
#include "SimpleNeuronBlock.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace tbb;

namespace NeuralNetNative {
	SimpleNeuronBlock::SimpleNeuronBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function) : BaseNeuralBlock(size, parent, function) {
	}

	SimpleNeuronBlock::SimpleNeuronBlock(int size, int parentSize, ActivationFunction *function) : BaseNeuralBlock(size, parentSize, function) {
	}

	void SimpleNeuronBlock::Calculate(void) {
		float *parentState = Parent->GetState();
		int parentSize = Parent->GetSize();
		
		parallel_for(blocked_range<size_t>(0, Size, SimpleNeuronBlockGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int neuronNum = r.begin(); neuronNum < r.end(); neuronNum++) {
				float sum = 0.0f;
				#pragma simd
				for (int i = 0; i < parentSize; i++) {
					sum += parentState[i]*Weights[neuronNum*parentSize + i];
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = Function->Calculate(sum);
			}
		});
	}

	void SimpleNeuronBlock::Calculate(const float *input) {
		parallel_for(blocked_range<size_t>(0, Size, SimpleNeuronBlockGrainSize),
		[=](const blocked_range<size_t>& r)
		{
			for (int neuronNum = r.begin(); neuronNum < r.end(); neuronNum++) {
				float sum = 0.0f;
				#pragma simd
				for (int i = 0; i < PreviousSize; i++) {
					sum += input[i]*Weights[neuronNum*PreviousSize + i];
				}
				Net[neuronNum] = sum + Bias[neuronNum];
				State[neuronNum] = Function->Calculate(sum);
			}
		});
	}
}