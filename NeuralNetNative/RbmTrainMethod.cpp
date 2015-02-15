#define NEURALNETNATIVEAPI

#include <mathimf.h>
#include "RbmTrainMethod.h"
#include <cfloat>

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
        RbmTrainMethod::RbmTrainMethod(StandardTypesNative::TrainSingle **trainData,
                       int trainDataSize,
                       GradientFunction *gradientFunction) {
            _trainDataIterator = new StandardTypesNative::RandomAccessIterator<StandardTypesNative::
                                                          TrainSingle*>(trainData, trainDataSize);
            _neuronNetOutput = (float*)_mm_malloc(trainData[0]->InputLength()*sizeof(float), 32);
			_testData = 0;
			_testDataSize = 0;

            _gradientFunction = gradientFunction;
        }

        RbmTrainMethod::RbmTrainMethod(StandardTypesNative::TrainSingle **trainData,
                       StandardTypesNative::TrainSingle **testData,
                       int trainDataSize, int testDataSize,
                       GradientFunction *gradientFunction) {
            _trainDataIterator = new StandardTypesNative::RandomAccessIterator<StandardTypesNative::
                                                          TrainSingle*>(trainData, trainDataSize);
            _neuronNetOutput = (float*)_mm_malloc(trainData[0]->InputLength()*sizeof(float), 32);
			_testData = testData;
			_testDataSize = testDataSize;

            _gradientFunction = gradientFunction;
        }

        RbmTrainMethod::~RbmTrainMethod(void) {
            _mm_free(_neuronNetOutput);
            
            delete _trainDataIterator;
            
            if (gradients != 0) {
                delete gradients;
                gradients = 0;
            }
        }

        void RbmTrainMethod::RunIterativeProcess(void) {
            float trainErrorValue = properties->Epsilon + 1.0f;
			float testErrorValue = std::numeric_limits<float>::quiet_NaN();
			float minTestErrorValue = FLT_MAX;
			epochNumber = 1;
			while ((ProcessSate == StandardTypesNative::IterativeProcessState::InProgress) && 
				(trainErrorValue > properties->Epsilon) && 
				(epochNumber <= properties->MaxIterationCount) &&
				(isnanf(testErrorValue) || fabsf(testErrorValue - minTestErrorValue) < properties->CvLimit)) {

				TrainEpoch();

				trainErrorValue = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());
				testErrorValue = TestModel(_testData, _testDataSize);
				if (!isnanf(testErrorValue) && (testErrorValue < minTestErrorValue)) {
					minTestErrorValue = testErrorValue;
				}

				OnIterationCompleted(epochNumber, trainErrorValue, testErrorValue);
				epochNumber++;
			}
			OnIterativeProcessFinished(epochNumber);
        }

        void RbmTrainMethod::ApplyResults(void) {
            if (ProcessSate == StandardTypesNative::IterativeProcessState::Finished) {
				DeleteTemporaryData();

                if (gradients != 0) {
                    delete gradients;
                    gradients = 0;
                }
			}
        }

        int RbmTrainMethod::CalculatePackagesCount(void) const {
            int count = _trainDataIterator->Size()/properties->PackageSize;
			if (_trainDataIterator->Size()%properties->PackageSize != 0) {
				count++;
			}
			return count;
        }

        float RbmTrainMethod::TestModel(StandardTypesNative::TrainSingle **data, int dataSize) const {
            if (data == 0 || dataSize == 0) {
	            return std::numeric_limits<float>::quiet_NaN();
	        }
	        else {
                float sumError = 0.0f;
                for (int i = 0; i < dataSize; i++) {
                    StandardTypesNative::TrainSingle *testExample = data[i];
                    neuralNet->Predict(testExample->Input(), _neuronNetOutput);
                    sumError += properties->Metrics->Calculate(testExample->Input(), _neuronNetOutput, visibleStatesCount);
                }
                return sumError / dataSize;
	        }
        }

        void RbmTrainMethod::TrainEpoch(void) {
            _trainDataIterator->RefreshRandomAccess();
			for (int i = 0; i < packagesCount; i++) {
				TrainPackage(i);
			}
        }

        void RbmTrainMethod::TrainPackage(int packageId) {
            _gradientFunction->PrepareToNextPackage(properties->PackageSize);
			for (int i = 0; i < properties->PackageSize; i++) {
				float *input = _trainDataIterator->Next()->Input();

				MakePositivePhase(input);
                _gradientFunction->StorePositivePhaseData(input, neuralNet->GetHiddenStates());
				MakeNegativePhase(packageId);
				_gradientFunction->StoreNegativePhaseData(GetVisibleStatesOnNegativePhase(packageId), GetHiddenStatesOnNegativePhase());
				RestoreVisibleStates(packageId);
			}
			_gradientFunction->MakeGradient(_packageFactor);
			ModifyWeightsOfNeuronNet();
        }

        void RbmTrainMethod::InitilazeMethod(NeuralNet *newNeuralNet, TrainProperties *newProperties) {
			neuralNet = dynamic_cast<RestrictedBoltzmannMachineBase*>(newNeuralNet);
			if (neuralNet == 0) {
				return;
			}

            visibleStatesCount = neuralNet->GetVisibleStatesCount();
            hiddenStatesCount = neuralNet->GetHiddenStatesCount();

            gradients = new RbmGradients(visibleStatesCount, hiddenStatesCount);
			_gradientFunction->Initialize(gradients);
				
			properties = newProperties;
			_packageFactor = 1.0f/properties->PackageSize;
			packagesCount = CalculatePackagesCount();

			CreateTemporaryData();

			ProcessSate = StandardTypesNative::IterativeProcessState::NotStarted;
		}

		TrainProperties* RbmTrainMethod::Properties(void) const {
			return properties;
		}
	}
}
