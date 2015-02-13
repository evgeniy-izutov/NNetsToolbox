#define NEURALNETNATIVEAPI

#include <mathimf.h>
#include "FastPersistentContrastiveDivergence.h"
#include <cfloat>
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <algorithm>

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		FastPersistentContrastiveDivergence::
            FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, 
                                                int trainDataSize,
                                                GradientFunction *gradientFunction,
                                                float fastWeightsDecreaseFactor)
        : RbmTrainMethod(trainData, trainDataSize, gradientFunction) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
		}

		FastPersistentContrastiveDivergence::
            FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                                StandardTypesNative::TrainSingle **testData,
                                                int trainDataSize, int testDataSize,
                                                GradientFunction *gradientFunction,
                                                float fastWeightsDecreaseFactor)
            : RbmTrainMethod(trainData, testData, trainDataSize, testDataSize, gradientFunction) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
		}

		FastPersistentContrastiveDivergence::~FastPersistentContrastiveDivergence(void) {
			DeleteTemporaryData();
		}

		void FastPersistentContrastiveDivergence::CreateTemporaryData(void) {
			int weightsCount = visibleStatesCount*hiddenStatesCount;
			_oldDeltaRegularWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_fastWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			
            std::fill(_oldDeltaRegularWeights, _oldDeltaRegularWeights + weightsCount, 0.0f);
            std::fill(_fastWeights, _fastWeights + weightsCount, 0.0f);

			_oldDeltaRegularWeightsForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
			_fastWeightsForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
			
            std::fill(_oldDeltaRegularWeightsForVisibleBias, _oldDeltaRegularWeightsForVisibleBias + visibleStatesCount, 0.0f);
            std::fill(_fastWeightsForVisibleBias, _fastWeightsForVisibleBias + visibleStatesCount, 0.0f);

			_persistentVisibleStates = (float*)_mm_malloc(packagesCount*visibleStatesCount*sizeof(float), 32);
			std::fill(_persistentVisibleStates, _persistentVisibleStates + packagesCount*visibleStatesCount, 0.0f);

			_oldDeltaRegularWeightsForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);
			_fastWeightsForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);
			
            std::fill(_oldDeltaRegularWeightsForHiddenBias, _oldDeltaRegularWeightsForHiddenBias + hiddenStatesCount, 0.0f);
            std::fill(_fastWeightsForHiddenBias, _fastWeightsForHiddenBias + hiddenStatesCount, 0.0f);
		}

		void FastPersistentContrastiveDivergence::DeleteTemporaryData(void) {
			if (_persistentVisibleStates != 0) {
                _persistentVisibleStates = 0;

				_mm_free(_persistentVisibleStates);
				_mm_free(_fastWeights);
				_mm_free(_fastWeightsForVisibleBias);
				_mm_free(_fastWeightsForHiddenBias);
				_mm_free(_oldDeltaRegularWeights);
				_mm_free(_oldDeltaRegularWeightsForVisibleBias);
				_mm_free(_oldDeltaRegularWeightsForHiddenBias);
			}
		}

        void FastPersistentContrastiveDivergence::MakePositivePhase(float *input) {
            neuralNet->HiddenLayerCalculateActivity(input);
        }

        void FastPersistentContrastiveDivergence::MakeNegativePhase(int packageId) {
            float *persistentVisibleStates = &_persistentVisibleStates[packageId*visibleStatesCount];
            neuralNet->HiddenLayerCalculateActivity(persistentVisibleStates, _fastWeights, _fastWeightsForHiddenBias);
        }

        float* FastPersistentContrastiveDivergence::GetVisibleStatesOnNegativePhase(int packageId) {
            return &_persistentVisibleStates[packageId*visibleStatesCount];
        }

        float* FastPersistentContrastiveDivergence::GetHiddenStatesOnNegativePhase(void) {
            return neuralNet->GetHiddenStates();
        }

        void FastPersistentContrastiveDivergence::RestoreVisibleStates(int packageId) {
            float *persistentVisibleStates = &_persistentVisibleStates[packageId*visibleStatesCount];
            
            neuralNet->VisibleLayerCalculateActivity(_fastWeights, _fastWeightsForVisibleBias);
			neuralNet->VisibleLayerCopyTo(persistentVisibleStates);
        }

        void FastPersistentContrastiveDivergence::ModifyWeightsOfNeuronNet() {
			float regularizationFactorPerPackage = 1.0f/packagesCount;
			float curRegularLearnSpeed = properties->BaseLearnSpeed*properties->FactorStrategy->GetFactor(epochNumber);
			float curFastLearnSpeed = properties->BaseLearnSpeed*properties->AddedFactorStrategy->GetFactor(epochNumber);
			
			float *regularWeights = neuralNet->GetWeights();
            float *packageDerivativeForWeights = gradients->GetPackageDerivativeForWeights();
			parallel_for(blocked_range<size_t>(0, hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					for (int i = 0; i < visibleStatesCount; i++) {
						int weightIndex = j*visibleStatesCount + i;

						float partialDerivative = packageDerivativeForWeights[weightIndex];
						packageDerivativeForWeights[weightIndex] = 0.0f;
						
						float newDeltaRegularWeight = properties->Momentum*_oldDeltaRegularWeights[weightIndex] +
						    curRegularLearnSpeed*(partialDerivative - 
							regularizationFactorPerPackage*properties->Regularization->GetDerivative(regularWeights[weightIndex]));
						_oldDeltaRegularWeights[weightIndex] = newDeltaRegularWeight;
						regularWeights[weightIndex] += (1.0f + properties->Momentum)*newDeltaRegularWeight;

						_fastWeights[weightIndex] = _fastWeightsDecreaseFactor*_fastWeights[weightIndex] +
                                                    curFastLearnSpeed*partialDerivative;
					}
				}
			});

			float *regularVisibleStatesBias = neuralNet->GetVisibleStatesBias();
            float *packageDerivativeForVisibleBias = gradients->GetPackageDerivativeForVisibleBias();
			parallel_for(blocked_range<size_t>(0, visibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					float partialDerivativeForVisibleBias = packageDerivativeForVisibleBias[i];
					packageDerivativeForVisibleBias[i] = 0.0f;
					
					float newDeltaForRegularVisibleBias = curRegularLearnSpeed*partialDerivativeForVisibleBias +
					                                      properties->Momentum*_oldDeltaRegularWeightsForVisibleBias[i];
					_oldDeltaRegularWeightsForVisibleBias[i] = newDeltaForRegularVisibleBias;
					regularVisibleStatesBias[i] += (1.0f + properties->Momentum)*newDeltaForRegularVisibleBias;

					_fastWeightsForVisibleBias[i] = _fastWeightsDecreaseFactor*_fastWeightsForVisibleBias[i] + 
						curFastLearnSpeed*partialDerivativeForVisibleBias;
				}
			});		

			float *regularHiddenStatesBias = neuralNet->GetHiddenStatesBias();
            float *packageDerivativeForHiddenBias = gradients->GetPackageDerivativeForHiddenBias();
			parallel_for(blocked_range<size_t>(0, hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float partialDerivativeForHiddenBias = packageDerivativeForHiddenBias[j];
					packageDerivativeForHiddenBias[j] = 0.0f;
										
					float newDeltaForHiddenBias = curRegularLearnSpeed*partialDerivativeForHiddenBias +
					                              properties->Momentum*_oldDeltaRegularWeightsForHiddenBias[j];
					_oldDeltaRegularWeightsForHiddenBias[j] = newDeltaForHiddenBias;
					regularHiddenStatesBias[j] += (1.0f + properties->Momentum)*newDeltaForHiddenBias;

					_fastWeightsForHiddenBias[j] = _fastWeightsDecreaseFactor*_fastWeightsForHiddenBias[j] + 
						curFastLearnSpeed*partialDerivativeForHiddenBias;
				}
			});
		}
	}
}