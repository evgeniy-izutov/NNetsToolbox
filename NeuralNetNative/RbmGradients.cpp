#define NEURALNETNATIVEAPI

#include "RbmGradients.h"
#include <malloc.h>
#include <algorithm>

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        RbmGradients::RbmGradients(int visibleStatesCount, int hiddenStatesCount) {
            _visibleStatesCount = visibleStatesCount;
            _hiddenStatesCount = hiddenStatesCount;

            _packageDerivativeForWeights = (float*)_mm_malloc(visibleStatesCount*hiddenStatesCount*sizeof(float), 32);
            _packageDerivativeForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
            _packageDerivativeForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);

            std::fill(_packageDerivativeForWeights, _packageDerivativeForWeights + visibleStatesCount*hiddenStatesCount, 0.0f);
            std::fill(_packageDerivativeForVisibleBias, _packageDerivativeForVisibleBias + visibleStatesCount, 0.0f);
            std::fill(_packageDerivativeForHiddenBias, _packageDerivativeForHiddenBias + hiddenStatesCount, 0.0f);
        }
        
        RbmGradients::~RbmGradients(void) {
            _mm_free(_packageDerivativeForWeights);
            _mm_free(_packageDerivativeForVisibleBias);
            _mm_free(_packageDerivativeForHiddenBias);

            _visibleStatesCount = 0;
            _hiddenStatesCount = 0;
        }

        int RbmGradients::GetVisibleStatesCount(void) const {
            return _visibleStatesCount;
        }

        int RbmGradients::GetHiddenStatesCount(void) const {
            return _hiddenStatesCount;
        }

        float* RbmGradients::GetPackageDerivativeForWeights(void) const {
            return _packageDerivativeForWeights;
        }

        float* RbmGradients::GetPackageDerivativeForVisibleBias(void) const {
            return _packageDerivativeForVisibleBias;
        }

        float* RbmGradients::GetPackageDerivativeForHiddenBias(void) const {
            return _packageDerivativeForHiddenBias;
        }
    }
}
