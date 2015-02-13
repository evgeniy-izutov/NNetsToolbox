#pragma once

#include "ExportDll.h"
#include "GradientFunction.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        class NEURALNETNATIVE_EXPORT CenteredGradient : public GradientFunction {
        public:
            public:
                virtual void PrepareToNextPackage(int nextPackageSize);
                virtual void StorePositivePhaseData(float *visibleStates, float *hiddenStates);
                virtual void StoreNegativePhaseData(float *visibleStates, float *hiddenStates);
                virtual void MakeGradient(float packageFactor);
        };
    }
}
