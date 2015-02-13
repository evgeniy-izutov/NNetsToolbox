#define NEURALNETNATIVEAPI

#include "CenteredGradient.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        void CenteredGradient::PrepareToNextPackage(int nextPackageSize) {}
        
        void CenteredGradient::StorePositivePhaseData(float *visibleStates, float *hiddenStates) {
        }

        void CenteredGradient::StoreNegativePhaseData(float *visibleStates, float *hiddenStates) {
        }

        void CenteredGradient::MakeGradient(float packageFactor) {
        }
    }
}
