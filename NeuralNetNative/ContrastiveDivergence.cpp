#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "ContrastiveDivergence.h"
#include <cfloat>
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>

using namespace StandardTypesNative;
using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		ContrastiveDivergence::ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, int trainDataSize, int methodStepsCount) {
			_methodStepsCount = methodStepsCount;
			_trainDataIterator = new RandomAccessIterator<TrainSingle*>(trainData, trainDataSize);
			_testData = 0;
			_testDataSize = 0;
		}

		ContrastiveDivergence::ContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, StandardTypesNative::TrainSingle **testData, int trainDataSize, int testDataSize, int methodStepsCount) {
			_methodStepsCount = methodStepsCount;
			_trainDataIterator = new RandomAccessIterator<TrainSingle*>(trainData, trainDataSize);
			_testData = testData;
			_testDataSize = testDataSize;
		}

		ContrastiveDivergence::~ContrastiveDivergence(void) {
			delete _trainDataIterator;
			ClearData();
		}
		
		void ContrastiveDivergence::InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties) {
			_neuralNet = dynamic_cast<RestrictedBoltzmannMachineBase*>(neuralNet);
			if (_neuralNet == 0) {
				return;
			}
				
			_properties = trainProperties;
			_packageFactor = 1.0f/_properties->PackageSize;
			AllocateMemory();
			ProcessSate = IterativeProcessState::NotStarted;
		}

		TrainProperties* ContrastiveDivergence::Properties(void) {
			return _properties;
		}

		void ContrastiveDivergence::RunIterativeProcess(void) {
			int packagesCount = CalculatePackagesCount();
			float trainErrorValue = _properties->Epsilon + 1.0f;
			float testErrorValue = std::numeric_limits<float>::quiet_NaN();
			float minTestErrorValue = FLT_MAX;
			_epochNumber = 1;
			while ((ProcessSate == IterativeProcessState::InProgress) && 
				(trainErrorValue > _properties->Epsilon) && 
				(_epochNumber <= _properties->MaxIterationCount) &&
				(isnanf(testErrorValue) || fabsf(testErrorValue - minTestErrorValue) < _properties->CvLimit)) {

				TrainEpoch(packagesCount);

				trainErrorValue = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());
				testErrorValue = TestModel(_testData, _testDataSize);
				if (!isnanf(testErrorValue) && (testErrorValue < minTestErrorValue)) {
					minTestErrorValue = testErrorValue;
				}

				OnIterationCompleted(_epochNumber, trainErrorValue, testErrorValue);
				_epochNumber++;
			}
			OnIterativeProcessFinished(_epochNumber);
		}

		void ContrastiveDivergence::ApplyResults(void) {
			if (ProcessSate == IterativeProcessState::Finished) {
				ClearData();
			}
		}

		void ContrastiveDivergence::AllocateMemory(void) {
			_visibleStatesCount = _neuralNet->GetVisibleStatesCount();
			_hiddenStatesCount = _neuralNet->GetHiddenStatesCount();
			int weightsCount = _visibleStatesCount*_hiddenStatesCount;
			_oldDeltaWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_derivativeAverages = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_packageDerivative = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_learnFactors = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			for (int i = 0; i < weightsCount; i++) {
				_oldDeltaWeights[i] = 0.0f;
				_derivativeAverages[i] = 0.0f;
				_packageDerivative[i] = 0.0f;
				_learnFactors[i] = 1.0f;
			}

			_oldDeltaWeightsForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_derivativeAveragesForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_packageDerivativeForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_learnFactorsForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			for (int i = 0; i < _visibleStatesCount; i++) {
				_oldDeltaWeightsForVisibleBias[i] = 0.0f;
				_derivativeAveragesForVisibleBias[i] = 0.0f;
				_packageDerivativeForVisibleBias[i] = 0.0f;
				_learnFactorsForVisibleBias[i] = 1.0f;
			}

			_oldDeltaWeightsForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_derivativeAveragesForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_packageDerivativeForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_learnFactorsForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			for (int i = 0; i < _hiddenStatesCount; i++) {
				_oldDeltaWeightsForHiddenBias[i] = 0.0f;
				_derivativeAveragesForHiddenBias[i] = 0.0f;
				_packageDerivativeForHiddenBias[i] = 0.0f;
				_learnFactorsForHiddenBias[i] = 1.0f;
			}

			_neuronNetOutput = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
		}

		void ContrastiveDivergence::ClearData(void) {
			if (_neuralNet != 0) {
				_properties = 0;
				_neuralNet = 0;
					
				_mm_free(_oldDeltaWeights);
				_mm_free(_oldDeltaWeightsForVisibleBias);
				_mm_free(_oldDeltaWeightsForHiddenBias);
				_mm_free(_derivativeAverages);
				_mm_free(_packageDerivative);
				_mm_free(_derivativeAveragesForVisibleBias);
				_mm_free(_packageDerivativeForVisibleBias);
				_mm_free(_derivativeAveragesForHiddenBias);
				_mm_free(_packageDerivativeForHiddenBias);
				_mm_free(_learnFactors);
				_mm_free(_learnFactorsForVisibleBias);
				_mm_free(_learnFactorsForHiddenBias);
			}
		}

		int ContrastiveDivergence::CalculatePackagesCount(void) {
			int packagesCount = _trainDataIterator->Size()/_properties->PackageSize;
			if (_trainDataIterator->Size()%_properties->PackageSize != 0) {
				packagesCount++;
			}
			return packagesCount;
		}

		void ContrastiveDivergence::TrainEpoch(int packagesCount) {
			_trainDataIterator->RefreshRandomAccess();
			for (int i = 0; i < packagesCount; i++) {
				TrainPackage(1.0f/packagesCount);
			}
		}

		void ContrastiveDivergence::TrainPackage(float regularizationFactorPerPackage) {
			for (int i = 0; i < _properties->PackageSize; i++) {
				TrainSingle *trainSingle = _trainDataIterator->Next();
				float *neuronNetInput = trainSingle->Input();
				CollectWeightsDelta(neuronNetInput);
			}
			ModifyWeightsOfNeuronNet(regularizationFactorPerPackage);
		}

		void ContrastiveDivergence::CollectWeightsDelta(float *input) {
			MakePositivePhase(input);
			MakeSampling();
			MakeNegativePhase();
		}

		void ContrastiveDivergence::MakePositivePhase(float *input) {
			_neuralNet->HiddenLayerCalculateActivity(input);

			float *hiddenStates = _neuralNet->GetHiddenStates();
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*_visibleStatesCount;
					float hiddenState = hiddenStates[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						_packageDerivative[startIndex + i] += input[i]*hiddenState;
					}
				}
			});

			for (int i = 0; i < _visibleStatesCount; i++) {
				_packageDerivativeForVisibleBias[i] += input[i];
			}

			for (int j = 0; j < _hiddenStatesCount; j++) {
				_packageDerivativeForHiddenBias[j] += hiddenStates[j];
			}
			
			_neuralNet->HiddenLayerSampling();
			_neuralNet->VisibleLayerCalculateActivity();
		}

		void ContrastiveDivergence::MakeSampling(void) {
			for (int k = 1; k < _methodStepsCount; k++) {
				_neuralNet->HiddenLayerCalculateActivity();
				_neuralNet->HiddenLayerSampling();
				_neuralNet->VisibleLayerCalculateActivity();
			}
		}

		void ContrastiveDivergence::MakeNegativePhase(void) {
			_neuralNet->HiddenLayerCalculateActivity();

			float *visibleStates = _neuralNet->GetVisibleStates();
			float *hiddenStates = _neuralNet->GetHiddenStates();
			
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*_visibleStatesCount;
					float hiddenState = hiddenStates[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						_packageDerivative[startIndex + i] -= visibleStates[i]*hiddenState;
					}
				}
			});

			for (int i = 0; i < _visibleStatesCount; i++) {
				_packageDerivativeForVisibleBias[i] -= visibleStates[i];
			}

			for (int j = 0; j < _hiddenStatesCount; j++) {
				_packageDerivativeForHiddenBias[j] -= hiddenStates[j];
			}
		}

		void ContrastiveDivergence::ModifyWeightsOfNeuronNet(float regularizationFactorPerPackage) {
			float curLearnSpeed = _properties->BaseLearnSpeed*_properties->FactorStrategy->GetFactor(_epochNumber);
			float *weights = _neuralNet->GetWeights();
			
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					for (int i = 0; i < _visibleStatesCount; i++) {
						int weightIndex = j*_visibleStatesCount + i;

						float lastDerivativeAverage = _derivativeAverages[weightIndex];
						float partialDerivative = _packageFactor*_packageDerivative[weightIndex] - 
							regularizationFactorPerPackage*_properties->Regularization->GetDerivative(weights[weightIndex]);
						_packageDerivative[weightIndex] = 0.0f;
						_learnFactors[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
							fminf(_learnFactors[weightIndex] + _properties->SpeedBonus, _properties->SpeedUpBorder):
							fmaxf(_learnFactors[weightIndex]*_properties->SpeedPenalty, _properties->SpeedLowBorder);			
						_derivativeAverages[weightIndex] = _properties->AverageLearnFactor*partialDerivative + 
							(1.0f - _properties->AverageLearnFactor)*lastDerivativeAverage;

						float oldDeltaWeight = _properties->Momentum*_oldDeltaWeights[weightIndex];
						float newDeltaWeight = curLearnSpeed*_learnFactors[weightIndex]*partialDerivative + oldDeltaWeight;
						_oldDeltaWeights[weightIndex] = newDeltaWeight;
						weights[weightIndex] += newDeltaWeight + oldDeltaWeight;
					}
				}
			});

			float *visibleStatesBias = _neuralNet->GetVisibleStatesBias();
			parallel_for(blocked_range<size_t>(0, _visibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					float lastDerivativeAverageForVisibleBias = _derivativeAveragesForVisibleBias[i];
					float partialDerivativeForVisibleBias = _packageFactor*_packageDerivativeForVisibleBias[i];
					_packageDerivativeForVisibleBias[i] = 0.0f;
					_learnFactorsForVisibleBias[i] = (lastDerivativeAverageForVisibleBias*partialDerivativeForVisibleBias > 0.0f) ?
						fminf(_learnFactorsForVisibleBias[i] + _properties->SpeedBonus, _properties->SpeedUpBorder):
						fmaxf(_learnFactorsForVisibleBias[i]*_properties->SpeedPenalty, _properties->SpeedLowBorder);
					_derivativeAveragesForVisibleBias[i] = _properties->AverageLearnFactor*partialDerivativeForVisibleBias + 
						(1.0f - _properties->AverageLearnFactor)*lastDerivativeAverageForVisibleBias;

					float oldDeltaForVisibleBias = _properties->Momentum*_oldDeltaWeightsForVisibleBias[i];
					float newDeltaForVisibleBias = curLearnSpeed*_learnFactorsForVisibleBias[i]*partialDerivativeForVisibleBias + 
						oldDeltaForVisibleBias;
					_oldDeltaWeightsForVisibleBias[i] = newDeltaForVisibleBias;
					visibleStatesBias[i] += newDeltaForVisibleBias + oldDeltaForVisibleBias;
				}
			});		

			float *hiddenStatesBias = _neuralNet->GetHiddenStatesBias();
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float lastDerivativeAverageForHiddenBias = _derivativeAveragesForHiddenBias[j];
					float partialDerivativeForHiddenBias = _packageFactor*_packageDerivativeForHiddenBias[j];
					_packageDerivativeForHiddenBias[j] = 0.0f;
					_learnFactorsForHiddenBias[j] = (lastDerivativeAverageForHiddenBias*partialDerivativeForHiddenBias > 0.0f) ?
						fminf(_learnFactorsForHiddenBias[j] + _properties->SpeedBonus, _properties->SpeedUpBorder):
						fmaxf(_learnFactorsForHiddenBias[j]*_properties->SpeedPenalty, _properties->SpeedLowBorder);
					_derivativeAveragesForHiddenBias[j] = _properties->AverageLearnFactor*partialDerivativeForHiddenBias + 
						(1.0f - _properties->AverageLearnFactor)*lastDerivativeAverageForHiddenBias;
					
					float oldDeltaForHiddenBias = _properties->Momentum*_oldDeltaWeightsForHiddenBias[j];
					float newDeltaForHiddenBias = curLearnSpeed*_learnFactorsForHiddenBias[j]*partialDerivativeForHiddenBias + 
						oldDeltaForHiddenBias;
					_oldDeltaWeightsForHiddenBias[j] = newDeltaForHiddenBias;
					hiddenStatesBias[j] += newDeltaForHiddenBias + oldDeltaForHiddenBias;
				}
			});
		}

		float ContrastiveDivergence::TestModel(StandardTypesNative::TrainSingle **data, int dataSize) {
            if (data == 0 || dataSize == 0) {
	            return std::numeric_limits<float>::quiet_NaN();
	        }
	        else {
                float sumError = 0.0f;
                for (int i = 0; i < dataSize; i++) {
                    StandardTypesNative::TrainSingle *testExample = data[i];
                    _neuralNet->Predict(testExample->Input(), _neuronNetOutput);
                    sumError += _properties->Metrics->Calculate(testExample->Input(), _neuronNetOutput, _visibleStatesCount);
                }
                return sumError / dataSize;
	        }
	    }
	}
}