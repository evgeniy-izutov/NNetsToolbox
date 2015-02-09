#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "FastPersistentContrastiveDivergence.h"
#include <cfloat>
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>

using namespace StandardTypesNative;
using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		FastPersistentContrastiveDivergence::FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, int trainDataSize, float fastWeightsDecreaseFactor) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
			_trainDataIterator = new RandomAccessIterator<TrainSingle*>(trainData, trainDataSize);
			_testData = 0;
			_testDataSize = 0;
		}

		FastPersistentContrastiveDivergence::FastPersistentContrastiveDivergence(StandardTypesNative::TrainSingle **trainData, StandardTypesNative::TrainSingle **testData, int trainDataSize, int testDataSize, float fastWeightsDecreaseFactor) {
			_fastWeightsDecreaseFactor = fastWeightsDecreaseFactor;
			_trainDataIterator = new RandomAccessIterator<TrainSingle*>(trainData, trainDataSize);
			_testData = testData;
			_testDataSize = testDataSize;
		}

		FastPersistentContrastiveDivergence::~FastPersistentContrastiveDivergence(void) {
			delete _trainDataIterator;
			ClearData();
		}
		
		void FastPersistentContrastiveDivergence::InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties) {
			_neuralNet = dynamic_cast<RestrictedBoltzmannMachineBase*>(neuralNet);
			if (_neuralNet == 0) {
				return;
			}
				
			_properties = trainProperties;
			_packageFactor = 1.0f/_properties->PackageSize;
			_packagesCount = CalculatePackagesCount();
			AllocateMemory();
			ProcessSate = IterativeProcessState::NotStarted;
		}

		TrainProperties* FastPersistentContrastiveDivergence::Properties(void) {
			return _properties;
		}

		void FastPersistentContrastiveDivergence::RunIterativeProcess(void) {
			float trainErrorValue = _properties->Epsilon + 1.0f;
			float testErrorValue = std::numeric_limits<float>::quiet_NaN();
			float minTestErrorValue = FLT_MAX;
			_epochNumber = 1;
			while ((ProcessSate == IterativeProcessState::InProgress) && 
				(trainErrorValue > _properties->Epsilon) && 
				(_epochNumber <= _properties->MaxIterationCount) &&
				(isnanf(testErrorValue) || fabsf(testErrorValue - minTestErrorValue) < _properties->CvLimit)) {

				TrainEpoch();

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

		void FastPersistentContrastiveDivergence::ApplyResults(void) {
			if (ProcessSate == IterativeProcessState::Finished) {
				ClearData();
			}
		}

		void FastPersistentContrastiveDivergence::AllocateMemory(void) {
			_visibleStatesCount = _neuralNet->GetVisibleStatesCount();
			_hiddenStatesCount = _neuralNet->GetHiddenStatesCount();
			int weightsCount = _visibleStatesCount*_hiddenStatesCount;
			_oldDeltaRegularWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_packageDerivative = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			_fastWeights = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
			for (int i = 0; i < weightsCount; i++) {
				_oldDeltaRegularWeights[i] = 0.0f;
				_packageDerivative[i] = 0.0f;
				_fastWeights[i] = 0.0f;
			}

			_oldDeltaRegularWeightsForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_packageDerivativeForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_fastWeightsForVisibleBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			for (int i = 0; i < _visibleStatesCount; i++) {
				_oldDeltaRegularWeightsForVisibleBias[i] = 0.0f;
				_packageDerivativeForVisibleBias[i] = 0.0f;
				_fastWeightsForVisibleBias[i] = 0.0f;
			}

			_persistentVisibleStates = (float*)_mm_malloc(_packagesCount*_visibleStatesCount*sizeof(float), 32);
			for (int i = 0; i < _packagesCount*_visibleStatesCount; i++) {
				_persistentVisibleStates[i] = 0.0f;
			}

			_oldDeltaRegularWeightsForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_packageDerivativeForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_fastWeightsForHiddenBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			for (int i = 0; i < _hiddenStatesCount; i++) {
				_oldDeltaRegularWeightsForHiddenBias[i] = 0.0f;
				_packageDerivativeForHiddenBias[i] = 0.0f;
				_fastWeightsForHiddenBias[i] = 0.0f;
			}

			_neuronNetOutput = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
		}

		void FastPersistentContrastiveDivergence::ClearData(void) {
			if (_neuralNet != 0) {
				_properties = 0;
				_neuralNet = 0;
				
				_mm_free(_persistentVisibleStates);
				_mm_free(_fastWeights);
				_mm_free(_fastWeightsForVisibleBias);
				_mm_free(_fastWeightsForHiddenBias);
				_mm_free(_oldDeltaRegularWeights);
				_mm_free(_oldDeltaRegularWeightsForVisibleBias);
				_mm_free(_oldDeltaRegularWeightsForHiddenBias);
				_mm_free(_packageDerivative);
				_mm_free(_packageDerivativeForVisibleBias);
				_mm_free(_packageDerivativeForHiddenBias);
			}
		}

		int FastPersistentContrastiveDivergence::CalculatePackagesCount(void) {
			int packagesCount = _trainDataIterator->Size()/_properties->PackageSize;
			if (_trainDataIterator->Size()%_properties->PackageSize != 0) {
				packagesCount++;
			}
			return packagesCount;
		}

		void FastPersistentContrastiveDivergence::TrainEpoch() {
			_trainDataIterator->RefreshRandomAccess();
			for (int i = 0; i < _packagesCount; i++) {
				TrainPackage(i);
			}
		}

		void FastPersistentContrastiveDivergence::TrainPackage(int packageId) {
			for (int i = 0; i < _properties->PackageSize; i++) {
				MakePositivePhase(_trainDataIterator->Next()->Input());
				MakeNegativePhase(packageId);
			}
			ModifyWeightsOfNeuronNet();
		}

		void FastPersistentContrastiveDivergence::MakePositivePhase(float *input) {
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
					_packageDerivativeForHiddenBias[j] += hiddenState;
				}
			});

			for (int i = 0; i < _visibleStatesCount; i++) {
				_packageDerivativeForVisibleBias[i] += input[i];
			}
		}

		void FastPersistentContrastiveDivergence::MakeNegativePhase(int packageId) {
			float *persistentVisibleStates = &_persistentVisibleStates[packageId*_visibleStatesCount];
			_neuralNet->HiddenLayerCalculateActivity(persistentVisibleStates, _fastWeights, _fastWeightsForHiddenBias);

			float *hiddenStates = _neuralNet->GetHiddenStates();
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					int startIndex = j*_visibleStatesCount;
					float hiddenState = hiddenStates[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						_packageDerivative[startIndex + i] -= persistentVisibleStates[i]*hiddenState;
					}
					_packageDerivativeForHiddenBias[j] -= hiddenState;
				}
			});

			for (int i = 0; i < _visibleStatesCount; i++) {
				_packageDerivativeForVisibleBias[i] -= persistentVisibleStates[i];
			}

			_neuralNet->VisibleLayerCalculateActivity(_fastWeights, _fastWeightsForVisibleBias);
			_neuralNet->VisibleLayerCopyTo(persistentVisibleStates);

			//not to sample visible layer?
			//_neuralNet->VisibleLayerSampling(persistentVisibleStates);
		}

		void FastPersistentContrastiveDivergence::ModifyWeightsOfNeuronNet() {
			float regularizationFactorPerPackage = 1.0f/_packagesCount;
			float curRegularLearnSpeed = _properties->BaseLearnSpeed*_properties->FactorStrategy->GetFactor(_epochNumber);
			float curFastLearnSpeed = _properties->BaseLearnSpeed*_properties->AddedFactorStrategy->GetFactor(_epochNumber);
			
			float *regularWeights = _neuralNet->GetWeights();		
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					for (int i = 0; i < _visibleStatesCount; i++) {
						int weightIndex = j*_visibleStatesCount + i;

						float partialDerivative = _packageFactor*_packageDerivative[weightIndex];
						_packageDerivative[weightIndex] = 0.0f;
						
						float oldDeltaRegularWeight = _properties->Momentum*_oldDeltaRegularWeights[weightIndex];
						float newDeltaRegularWeight = oldDeltaRegularWeight + curRegularLearnSpeed*(partialDerivative - 
							regularizationFactorPerPackage*_properties->Regularization->GetDerivative(regularWeights[weightIndex]));
						_oldDeltaRegularWeights[weightIndex] = newDeltaRegularWeight;
						regularWeights[weightIndex] += newDeltaRegularWeight + oldDeltaRegularWeight;

						_fastWeights[weightIndex] = _fastWeightsDecreaseFactor*_fastWeights[weightIndex] + curFastLearnSpeed*partialDerivative;
					}
				}
			});

			float *regularVisibleStatesBias = _neuralNet->GetVisibleStatesBias();
			parallel_for(blocked_range<size_t>(0, _visibleStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					float partialDerivativeForVisibleBias = _packageFactor*_packageDerivativeForVisibleBias[i];
					_packageDerivativeForVisibleBias[i] = 0.0f;
					
					float oldDeltaForRegularVisibleBias = _properties->Momentum*_oldDeltaRegularWeightsForVisibleBias[i];
					float newDeltaForRegularVisibleBias = curRegularLearnSpeed*partialDerivativeForVisibleBias + oldDeltaForRegularVisibleBias;
					_oldDeltaRegularWeightsForVisibleBias[i] = newDeltaForRegularVisibleBias;
					regularVisibleStatesBias[i] += newDeltaForRegularVisibleBias + oldDeltaForRegularVisibleBias;

					_fastWeightsForVisibleBias[i] = _fastWeightsDecreaseFactor*_fastWeightsForVisibleBias[i] + 
						curFastLearnSpeed*partialDerivativeForVisibleBias;
				}
			});		

			float *regularHiddenStatesBias = _neuralNet->GetHiddenStatesBias();
			parallel_for(blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float partialDerivativeForHiddenBias = _packageFactor*_packageDerivativeForHiddenBias[j];
					_packageDerivativeForHiddenBias[j] = 0.0f;
										
					float oldDeltaForHiddenBias = _properties->Momentum*_oldDeltaRegularWeightsForHiddenBias[j];
					float newDeltaForHiddenBias = curRegularLearnSpeed*partialDerivativeForHiddenBias + oldDeltaForHiddenBias;
					_oldDeltaRegularWeightsForHiddenBias[j] = newDeltaForHiddenBias;
					regularHiddenStatesBias[j] += newDeltaForHiddenBias + oldDeltaForHiddenBias;

					_fastWeightsForHiddenBias[j] = _fastWeightsDecreaseFactor*_fastWeightsForHiddenBias[j] + 
						curFastLearnSpeed*partialDerivativeForHiddenBias;
				}
			});
		}

		float FastPersistentContrastiveDivergence::TestModel(StandardTypesNative::TrainSingle **data, int dataSize) {
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