#define NEURALNETNATIVEAPI

#include "CenteredGradient.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <algorithm>
#include <malloc.h>

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        CenteredGradient::CenteredGradient(float slidingFactor,
                                           float *visibleOffsets, int visibleOffsetsCount,
                                           float *hiddenOffsets, int hiddenOffsetsCount) {
            _slidingFactor = slidingFactor;
			
            _visibleOffsets = (float*)_mm_malloc(visibleOffsetsCount*sizeof(float), 32);
            std::copy(visibleOffsets, visibleOffsets + visibleOffsetsCount, _visibleOffsets);

            _hiddenOffsets = (float*)_mm_malloc(hiddenOffsetsCount*sizeof(float), 32);
            std::copy(hiddenOffsets, hiddenOffsets + hiddenOffsetsCount, _hiddenOffsets);

            _dataVisibleHidden = 0;
        }

        CenteredGradient::~CenteredGradient(void) {
            _mm_free(_visibleOffsets);
            _mm_free(_hiddenOffsets);

            DeleteMemory();
        }

        void CenteredGradient::AllocateMemory(void) {
            _dataVisibleHidden = (float*)_mm_malloc(VisibleStatesCount*HiddenStatesCount*sizeof(float), 32);
            _modelVisibleHidden = (float*)_mm_malloc(VisibleStatesCount*HiddenStatesCount*sizeof(float), 32);

            _dataVisible = (float*)_mm_malloc(VisibleStatesCount*sizeof(float), 32);
            _modelVisible = (float*)_mm_malloc(VisibleStatesCount*sizeof(float), 32);

            _dataHidden = (float*)_mm_malloc(HiddenStatesCount*sizeof(float), 32);
            _modelHidden = (float*)_mm_malloc(HiddenStatesCount*sizeof(float), 32);

            _visibleOffsetsNew = (float*)_mm_malloc(VisibleStatesCount*sizeof(float), 32);
            _hiddenOffsetsNew = (float*)_mm_malloc(HiddenStatesCount*sizeof(float), 32);
        }

        void CenteredGradient::DeleteMemory(void) {
            if (_dataVisibleHidden != 0) {
                _mm_free(_dataVisibleHidden);
                _mm_free(_modelVisibleHidden);
                _mm_free(_dataVisible);
                _mm_free(_modelVisible);
                _mm_free(_dataHidden);
                _mm_free(_modelHidden);
                _mm_free(_visibleOffsetsNew);
                _mm_free(_hiddenOffsetsNew);

                _dataVisibleHidden = 0;
            }
        }
        
        void CenteredGradient::PrepareToNextPackage(int nextPackageSize) {
            _packageFactor = 1.0f/nextPackageSize;

            std::fill(_visibleOffsetsNew, _visibleOffsetsNew + VisibleStatesCount, 0.0f);
            std::fill(_hiddenOffsetsNew, _hiddenOffsetsNew + HiddenStatesCount, 0.0f);
			
            std::fill(_dataVisibleHidden, _dataVisibleHidden + VisibleStatesCount*HiddenStatesCount, 0.0f);
            std::fill(_modelVisibleHidden, _modelVisibleHidden + VisibleStatesCount*HiddenStatesCount, 0.0f);

            std::fill(_dataHidden, _dataHidden + HiddenStatesCount, 0.0f);
            std::fill(_modelHidden, _modelHidden + HiddenStatesCount, 0.0f);

            std::fill(_dataVisible, _dataVisible + VisibleStatesCount, 0.0f);
            std::fill(_modelVisible, _modelVisible + VisibleStatesCount, 0.0f);
        }
        
        void CenteredGradient::StorePositivePhaseData(float *visibleStates, float *hiddenStates) {
            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
               for (int j = r.begin(); j < r.end(); j++) {
			   	   float shiftedHiddenState = hiddenStates[j] - _hiddenOffsets[j];
			   	   for (int i = 0; i < VisibleStatesCount; i++) {
			   	   	   _dataVisibleHidden[j*VisibleStatesCount + i] += (visibleStates[i] - _visibleOffsets[i])*shiftedHiddenState;
			   	   }
			   	   _dataHidden[j] += hiddenStates[j];
			   	   _hiddenOffsetsNew[j] += _packageFactor*hiddenStates[j];
			   }
            });

            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for (int i = r.begin(); i < r.end(); i++) {
			    	_dataVisible[i] += visibleStates[i];
			    	_visibleOffsetsNew[i] += _packageFactor*visibleStates[i];
			    }
            });
        }

        void CenteredGradient::StoreNegativePhaseData(float *visibleStates, float *hiddenStates) {
            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for (int j = r.begin(); j < r.end(); j++) {
			    	float shiftedHiddenState = hiddenStates[j] - _hiddenOffsets[j];
			    	for (int i = 0; i < VisibleStatesCount; i++) {
			    		_modelVisibleHidden[j*VisibleStatesCount + i] += (visibleStates[i] - _visibleOffsets[i])*shiftedHiddenState;
			    	}
			    	_modelHidden[j] += hiddenStates[j];
			    }
            });
            
            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for (int i = r.begin(); i < r.end(); i++) {
			    	_modelVisible[i] += visibleStates[i];
			    }
            });
        }

        void CenteredGradient::MakeGradient(float packageFactor) {
            float *packageDerivativeForWeights = Gradients->GetPackageDerivativeForWeights();
            float *packageDerivativeForHiddenBias = Gradients->GetPackageDerivativeForHiddenBias();
            float *packageDerivativeForVisibleBias = Gradients->GetPackageDerivativeForVisibleBias();
            
            parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for(int j = r.begin(); j < r.end(); j++) {
			    	int startIndex = j*VisibleStatesCount;
                    for (int i = 0; i < VisibleStatesCount; i++) {
			    		packageDerivativeForWeights[startIndex + i] = 
                            packageFactor*(_dataVisibleHidden[startIndex + i] - _modelVisibleHidden[startIndex + i]);
			    	}
			    }
            });

            parallel_for( blocked_range<size_t>(0, VisibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for (int i = r.begin(); i < r.end(); i++) {
			    	float visibleStateSumGradient = 0.0f;
			    	for (int j = 0; j < HiddenStatesCount; j++) {
			    		visibleStateSumGradient += _hiddenOffsets[j]*packageDerivativeForWeights[j*VisibleStatesCount + i];
			    	}
			    	packageDerivativeForVisibleBias[i] = packageFactor*(_dataVisible[i] - _modelVisible[i]) -
			    	                                     visibleStateSumGradient;
                
                    _visibleOffsets[i] = (1.0f - _slidingFactor)*_visibleOffsets[i] +
			    	                     _slidingFactor*_visibleOffsetsNew[i];
			    }
            });

			parallel_for( blocked_range<size_t>(0, HiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
                for (int j = r.begin(); j < r.end(); j++) {
			        float hiddenStateSumGradient = 0.0f;
                    for (int i = 0; i < VisibleStatesCount; i++) {
			    		hiddenStateSumGradient += _visibleOffsets[i]*packageDerivativeForWeights[j*VisibleStatesCount + i];
			    	}
			    	packageDerivativeForHiddenBias[j] = packageFactor*(_dataHidden[j] - _modelHidden[j]) - 
			    	                                    hiddenStateSumGradient;
                    
                    _hiddenOffsets[j] = (1.0f - _slidingFactor)*_hiddenOffsets[j] +
			    	                    _slidingFactor*_hiddenOffsetsNew[j];
			    }
            });
        }
    }
}
