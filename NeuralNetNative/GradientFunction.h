#pragma once
#include "RbmGradients.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        class GradientFunction {
        public:
            void Initialize(RbmGradients *gradients);
            virtual void StorePositivePhaseData(float *visibleStates, float *hiddenStates) = 0;
            virtual void StoreNegativePhaseData(float *visibleStates, float *hiddenStates) = 0;
            virtual void MakeGradient(float packageFactor) = 0;
        protected:
            RbmGradients *Gradients;
            int VisibleStatesCount;
            int HiddenStatesCount;
        protected:
            virtual void AllocateMemory(void);
            virtual void DeleteMemory(void);
        };
    }
}
