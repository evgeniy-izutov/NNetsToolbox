#pragma once

#include "ExportDll.h"
#include "GradientFunction.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        class NEURALNETNATIVE_EXPORT CenteredGradient : public GradientFunction {
        public:
            CenteredGradient(float slidingFactor,
                             float *visibleOffsets, int visibleOffsetsCount,
                             float *hiddenOffsets, int hiddenOffsetsCount);
            ~CenteredGradient(void);
            virtual void PrepareToNextPackage(int nextPackageSize);
            virtual void StorePositivePhaseData(float *visibleStates, float *hiddenStates);
            virtual void StoreNegativePhaseData(float *visibleStates, float *hiddenStates);
            virtual void MakeGradient(float packageFactor);
        protected:
            virtual void AllocateMemory(void);
            virtual void DeleteMemory(void);
        private:
            float _slidingFactor;
		    float *_visibleOffsets;
		    float *_hiddenOffsets;
		    float *_visibleOffsetsNew;
		    float *_hiddenOffsetsNew;
		    float *_dataVisibleHidden;
		    float *_dataVisible;
		    float *_dataHidden;
		    float *_modelVisibleHidden;
		    float *_modelVisible;
		    float *_modelHidden;
		    float _packageFactor;
        };
    }
}
