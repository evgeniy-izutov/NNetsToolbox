#pragma once

#include "ExportDll.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        class NEURALNETNATIVE_EXPORT RbmGradients {
        public:
	        RbmGradients(int visibleStatesCount, int hiddenStatesCount);
	        ~RbmGradients(void);
            int GetVisibleStatesCount(void) const;
            int GetHiddenStatesCount(void) const;
            float* GetPackageDerivativeForWeights(void) const;
            float* GetPackageDerivativeForVisibleBias(void) const;
            float* GetPackageDerivativeForHiddenBias(void) const;
        private:
            int _visibleStatesCount;
            int _hiddenStatesCount;
            float *_packageDerivativeForWeights;
            float *_packageDerivativeForVisibleBias;
            float *_packageDerivativeForHiddenBias;
        };
    }
}
