#include "GradientFunction.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        void GradientFunction::Initialize(RbmGradients *gradients) {
            Gradients = gradients;

            VisibleStatesCount = gradients->GetVisibleStatesCount();
            HiddenStatesCount = gradients->GetHiddenStatesCount();

            DeleteMemory();
            AllocateMemory();
        }

        void GradientFunction::AllocateMemory(void) {}

        void GradientFunction::DeleteMemory(void) {}
    }
}