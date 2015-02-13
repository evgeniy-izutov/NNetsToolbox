#define NEURALNETNATIVEAPI

#include <mathimf.h>
#include "ContrastiveDivergence.h"
#include <cfloat>
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <algorithm>

using namespace StandardTypesNative;
using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		ContrastiveDivergence::
            ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                  int trainDataSize,
                                  GradientFunction *gradientFunction,
                                  int methodStepsCount)
            : RbmTrainMethod(trainData, trainDataSize, gradientFunction) {
			_methodStepsCount = methodStepsCount;
		}

		ContrastiveDivergence::
            ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData,
                                  StandardTypesNative::TrainSingle **testData,
                                  int trainDataSize, int testDataSize,
                                  GradientFunction *gradientFunction,
                                  int methodStepsCount) 
            : RbmTrainMethod(trainData, testData, trainDataSize, testDataSize, gradientFunction) {
			_methodStepsCount = methodStepsCount;
		}

		ContrastiveDivergence::~ContrastiveDivergence(void) {
			DeleteTemporaryData();
		}

		void ContrastiveDivergence::CreateTemporaryData(void) {
			int weightsCount = visibleStatesCount*hiddenStatesCount;
			_oldDeltaWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_derivativeAverages = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_learnFactors = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			
            std::fill(_oldDeltaWeights, _oldDeltaWeights + weightsCount, 0.0f);
            std::fill(_derivativeAverages, _derivativeAverages + weightsCount, 0.0f);
            std::fill(_learnFactors, _learnFactors + weightsCount, 1.0f);

			_oldDeltaWeightsForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
			_derivativeAveragesForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
			_learnFactorsForVisibleBias = (float*)_mm_malloc(visibleStatesCount*sizeof(float), 32);
			
            std::fill(_oldDeltaWeightsForVisibleBias, _oldDeltaWeightsForVisibleBias + visibleStatesCount, 0.0f);
            std::fill(_derivativeAveragesForVisibleBias, _derivativeAveragesForVisibleBias + visibleStatesCount, 0.0f);
            std::fill(_learnFactorsForVisibleBias, _learnFactorsForVisibleBias + visibleStatesCount, 1.0f);

			_oldDeltaWeightsForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);
			_derivativeAveragesForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);
			_learnFactorsForHiddenBias = (float*)_mm_malloc(hiddenStatesCount*sizeof(float), 32);
			
            std::fill(_oldDeltaWeightsForHiddenBias, _oldDeltaWeightsForHiddenBias + hiddenStatesCount, 0.0f);
            std::fill(_derivativeAveragesForHiddenBias, _derivativeAveragesForHiddenBias + hiddenStatesCount, 0.0f);
            std::fill(_learnFactorsForHiddenBias, _learnFactorsForHiddenBias + hiddenStatesCount, 1.0f);
		}

		void ContrastiveDivergence::DeleteTemporaryData(void) {
			if (neuralNet != 0) {
				_oldDeltaWeights = 0;
                
                _mm_free(_oldDeltaWeights);
				_mm_free(_oldDeltaWeightsForVisibleBias);
				_mm_free(_oldDeltaWeightsForHiddenBias);
				_mm_free(_derivativeAverages);
				_mm_free(_derivativeAveragesForVisibleBias);
				_mm_free(_derivativeAveragesForHiddenBias);
				_mm_free(_learnFactors);
				_mm_free(_learnFactorsForVisibleBias);
				_mm_free(_learnFactorsForHiddenBias);
			}
		}

        void ContrastiveDivergence::MakePositivePhase(float *input) {
            neuralNet->HiddenLayerCalculateActivity(input);
        }

        void ContrastiveDivergence::MakeNegativePhase(int packageId) {
            neuralNet->HiddenLayerSampling();
			neuralNet->VisibleLayerCalculateActivity();
			for (int k = 1; k < _methodStepsCount; k++) {
				neuralNet->HiddenLayerCalculateActivity();
				neuralNet->HiddenLayerSampling();
				neuralNet->VisibleLayerCalculateActivity();
			}
			neuralNet->HiddenLayerCalculateActivity();
        }

        float* ContrastiveDivergence::GetVisibleStatesOnNegativePhase(int packageId) {
            return neuralNet->GetVisibleStates();
        }

        float* ContrastiveDivergence::GetHiddenStatesOnNegativePhase(void) {
            return neuralNet->GetHiddenStates();
        }

        void ContrastiveDivergence::RestoreVisibleStates(int packageId) {
        }

		void ContrastiveDivergence::ModifyWeightsOfNeuronNet() {
			float regularizationFactorPerPackage = 1.0f/packagesCount;
            float curLearnSpeed = properties->BaseLearnSpeed*properties->FactorStrategy->GetFactor(epochNumber);
			
            float *weights = neuralNet->GetWeights();
			float *packageDerivativeForWeights = gradients->GetPackageDerivativeForWeights();
			parallel_for(blocked_range<size_t>(0, hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					for (int i = 0; i < visibleStatesCount; i++) {
						int weightIndex = j*visibleStatesCount + i;

						float lastDerivativeAverage = _derivativeAverages[weightIndex];
						float partialDerivative = packageDerivativeForWeights[weightIndex] - 
							regularizationFactorPerPackage*properties->Regularization->GetDerivative(weights[weightIndex]);
						packageDerivativeForWeights[weightIndex] = 0.0f;
						_learnFactors[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
							fminf(_learnFactors[weightIndex] + properties->SpeedBonus, properties->SpeedUpBorder):
							fmaxf(_learnFactors[weightIndex]*properties->SpeedPenalty, properties->SpeedLowBorder);			
						_derivativeAverages[weightIndex] = properties->AverageLearnFactor*partialDerivative + 
							(1.0f - properties->AverageLearnFactor)*lastDerivativeAverage;

						float newDeltaWeight = curLearnSpeed*_learnFactors[weightIndex]*partialDerivative + 
						                       properties->Momentum*_oldDeltaWeights[weightIndex];
						_oldDeltaWeights[weightIndex] = newDeltaWeight;
						weights[weightIndex] += (1.0f + properties->Momentum)*newDeltaWeight;
					}
				}
			});

			float *visibleStatesBias = neuralNet->GetVisibleStatesBias();
            float *packageDerivativeForVisibleBias = gradients->GetPackageDerivativeForVisibleBias();
			parallel_for(blocked_range<size_t>(0, visibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					float lastDerivativeAverageForVisibleBias = _derivativeAveragesForVisibleBias[i];
					float partialDerivativeForVisibleBias = packageDerivativeForVisibleBias[i];
					packageDerivativeForVisibleBias[i] = 0.0f;
					_learnFactorsForVisibleBias[i] = (lastDerivativeAverageForVisibleBias*partialDerivativeForVisibleBias > 0.0f) ?
						fminf(_learnFactorsForVisibleBias[i] + properties->SpeedBonus, properties->SpeedUpBorder):
						fmaxf(_learnFactorsForVisibleBias[i]*properties->SpeedPenalty, properties->SpeedLowBorder);
					_derivativeAveragesForVisibleBias[i] = properties->AverageLearnFactor*partialDerivativeForVisibleBias + 
						(1.0f - properties->AverageLearnFactor)*lastDerivativeAverageForVisibleBias;

					float newDeltaForVisibleBias = curLearnSpeed*_learnFactorsForVisibleBias[i]*partialDerivativeForVisibleBias + 
						                           properties->Momentum*_oldDeltaWeightsForVisibleBias[i];
					_oldDeltaWeightsForVisibleBias[i] = newDeltaForVisibleBias;
					visibleStatesBias[i] += (1.0f + properties->Momentum)*newDeltaForVisibleBias;
				}
			});		

			float *hiddenStatesBias = neuralNet->GetHiddenStatesBias();
            float *packageDerivativeForHiddenBias = gradients->GetPackageDerivativeForHiddenBias();
			parallel_for(blocked_range<size_t>(0, hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float lastDerivativeAverageForHiddenBias = _derivativeAveragesForHiddenBias[j];
					float partialDerivativeForHiddenBias = packageDerivativeForHiddenBias[j] - 
							regularizationFactorPerPackage*properties->Regularization->GetDerivative(hiddenStatesBias[j]);
					packageDerivativeForHiddenBias[j] = 0.0f;
					_learnFactorsForHiddenBias[j] = (lastDerivativeAverageForHiddenBias*partialDerivativeForHiddenBias > 0.0f) ?
						fminf(_learnFactorsForHiddenBias[j] + properties->SpeedBonus, properties->SpeedUpBorder):
						fmaxf(_learnFactorsForHiddenBias[j]*properties->SpeedPenalty, properties->SpeedLowBorder);
					_derivativeAveragesForHiddenBias[j] = properties->AverageLearnFactor*partialDerivativeForHiddenBias + 
						(1.0f - properties->AverageLearnFactor)*lastDerivativeAverageForHiddenBias;
					
					float newDeltaForHiddenBias = curLearnSpeed*_learnFactorsForHiddenBias[j]*partialDerivativeForHiddenBias + 
						                          properties->Momentum*_oldDeltaWeightsForHiddenBias[j];
					_oldDeltaWeightsForHiddenBias[j] = newDeltaForHiddenBias;
					hiddenStatesBias[j] += (1.0f + properties->Momentum)*newDeltaForHiddenBias;
				}
			});
		}
	}
}