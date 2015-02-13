#define NEURALNETNATIVEAPI

#include "LinearGradient.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        void LinearGradient::PrepareToNextPackage(int nextPackageSize) {}
        
        void LinearGradient::StorePositivePhaseData(float *visibleStates, float *hiddenStates) {
            float *packageDerivativeForWeights = Gradients->GetPackageDerivativeForWeights();
            float *packageDerivativeForHiddenBias = Gradients->GetPackageDerivativeForHiddenBias();
            float *packageDerivativeForVisibleBias = Gradients->GetPackageDerivativeForVisibleBias();
            
            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*VisibleStatesCount;
				    int hiddenSate = hiddenStates[j];
				    for (int i = 0; i < VisibleStatesCount; i++) {
				    	packageDerivativeForWeights[startIndex + i] += visibleStates[i]*hiddenSate;
				    }
				    packageDerivativeForHiddenBias[j] += hiddenSate;
				}
			});

            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					packageDerivativeForVisibleBias[i] += visibleStates[i];
				}
			});
        }

        void LinearGradient::StoreNegativePhaseData(float *visibleStates, float *hiddenStates) {
            float *packageDerivativeForWeights = Gradients->GetPackageDerivativeForWeights();
            float *packageDerivativeForHiddenBias = Gradients->GetPackageDerivativeForHiddenBias();
            float *packageDerivativeForVisibleBias = Gradients->GetPackageDerivativeForVisibleBias();
            
            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*VisibleStatesCount;
				    int hiddenSate = hiddenStates[j];
				    for (int i = 0; i < VisibleStatesCount; i++) {
				    	packageDerivativeForWeights[startIndex + i] -= visibleStates[i]*hiddenSate;
				    }
				    packageDerivativeForHiddenBias[j] -= hiddenSate;
				}
			});

            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					packageDerivativeForVisibleBias[i] -= visibleStates[i];
				}
			});
        }

        void LinearGradient::MakeGradient(float packageFactor) {
            float *packageDerivativeForWeights = Gradients->GetPackageDerivativeForWeights();
            float *packageDerivativeForHiddenBias = Gradients->GetPackageDerivativeForHiddenBias();
            float *packageDerivativeForVisibleBias = Gradients->GetPackageDerivativeForVisibleBias();

            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*VisibleStatesCount;
				    for (int i = 0; i < VisibleStatesCount; i++) {
				    	packageDerivativeForWeights[startIndex + i] *= packageFactor;
				    }
				    packageDerivativeForHiddenBias[j] *= packageFactor;
				}
			});

            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					packageDerivativeForVisibleBias[i] *= packageFactor;
				}
			});
        }
    }
}
