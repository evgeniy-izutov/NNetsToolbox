#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "SoftmaxNeuronBlock.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_reduce.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace tbb;

namespace NeuralNetNative {
	SoftmaxSimpleNeuronBlock::SoftmaxSimpleNeuronBlock(int size, BaseNeuralBlock *parent, ActivationFunction *function) : BaseNeuralBlock(size, parent, function) {
	}

	SoftmaxSimpleNeuronBlock::SoftmaxSimpleNeuronBlock(int size, int parentSize, ActivationFunction *function) : BaseNeuralBlock(size, parentSize, function) {
	}

	void SoftmaxSimpleNeuronBlock::Calculate(void) {
		float *parentState = Parent->GetState();
		int parentSize = Parent->GetSize();
		float expSum = parallel_reduce(blocked_range<size_t>(0, Size, SoftmaxNeuronBlockGrainSize), 
			0.0f, 
			[=](const blocked_range<size_t>& r, float sum)->float 
			{
				for (int neuronNum = r.begin(); neuronNum < r.end(); neuronNum++) {
					float inductionSum = 0.0f;
					#pragma simd
					for (int i = 0; i < parentSize; i++) {
						inductionSum += parentState[i]*Weights[neuronNum*parentSize + i];
					}
					inductionSum += Bias[neuronNum];
					Net[neuronNum] = inductionSum;
					float expValue = expf(inductionSum);
					State[neuronNum] = expValue;
					sum += expValue;
				}
				return sum;
			},
			[](float x, float y)->float 
			{
				return x+y;
			}
		);

		for (int neuronNum = 0; neuronNum < Size; neuronNum++) {
			State[neuronNum] = State[neuronNum]/expSum;
		}
	}

	void SoftmaxSimpleNeuronBlock::Calculate(const float *input) {
		float expSum = parallel_reduce(blocked_range<size_t>(0, Size, SoftmaxNeuronBlockGrainSize), 
			0.0f, 
			[=](const blocked_range<size_t>& r, float sum)->float 
			{
				for (int neuronNum = 0; neuronNum < Size; neuronNum++) {
					float inductionSum = 0.0f;
					#pragma simd
					for (int i = 0; i < PreviousSize; i++) {
						inductionSum += input[i]*Weights[neuronNum*PreviousSize + i];
					}
					inductionSum += Bias[neuronNum];
					Net[neuronNum] = inductionSum;
					float expValue = expf(inductionSum);
					State[neuronNum] = expValue;
					sum += expValue;
				}
				return sum;
			},
			[](float x, float y)->float 
			{
				return x+y;
			}
		);
		
		for (int neuronNum = 0; neuronNum < Size; neuronNum++) {
			State[neuronNum] = State[neuronNum]/expSum;
		}
	}
}