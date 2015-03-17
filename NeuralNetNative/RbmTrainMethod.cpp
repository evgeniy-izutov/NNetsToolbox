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
            if (IsTestDataAvailable()) {
	            RunTraingWithTesting();
	        }
			else {
				RunTraingWithoutTesting();
			}
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

        bool RbmTrainMethod::IsTestDataAvailable() const {
            return !(_testData == 0 || _testDataSize == 0);
        }

        void RbmTrainMethod::RunTraingWithTesting(void) {
            float trainError = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());
			float slidingTestError = TestModel(_testData, _testDataSize);
			float minTestError = slidingTestError;
			epochNumber = 1;
			while ((ProcessSate == StandardTypesNative::IterativeProcessState::InProgress) && 
				(trainError > properties->Epsilon) && 
				(epochNumber <= properties->MaxIterationCount) &&
				((epochNumber <= properties->SkipCvLimitFirstIterations) ||
                 (fabsf(slidingTestError - minTestError) < properties->CvLimit))) {

				TrainEpoch();

				trainError = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());
				float testError = TestModel(_testData, _testDataSize);
                slidingTestError = properties->CvSlidingFactor*testError +
					(1.0f - properties->CvSlidingFactor)*slidingTestError;

				if (testError < minTestError) {
					minTestError = testError;
				}

				OnIterationCompleted(epochNumber, trainError, testError);
				epochNumber++;
			}
			OnIterativeProcessFinished(epochNumber);
        }

        void RbmTrainMethod::RunTraingWithoutTesting(void) {
            float trainError = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());
			epochNumber = 1;
			while ((ProcessSate == StandardTypesNative::IterativeProcessState::InProgress) && 
				   (trainError > properties->Epsilon) && 
				   (epochNumber <= properties->MaxIterationCount)) {

				TrainEpoch();

				trainError = TestModel(_trainDataIterator->Collection(), _trainDataIterator->Size());

				OnIterationCompleted(epochNumber, trainError, std::numeric_limits<float>::quiet_NaN());
				epochNumber++;
			}
			OnIterativeProcessFinished(epochNumber);
        }

        int RbmTrainMethod::CalculatePackagesCount(void) const {
            int count = _trainDataIterator->Size()/properties->PackageSize;
			if (_trainDataIterator->Size()%properties->PackageSize != 0) {
				count++;
			}
			return count;
        }

        float RbmTrainMethod::TestModel(StandardTypesNative::TrainSingle **data, int dataSize) const {
            float sumError = 0.0f;
            for (int i = 0; i < dataSize; i++) {
                StandardTypesNative::TrainSingle *testExample = data[i];
                neuralNet->Predict(testExample->Input(), _neuronNetOutput);
                sumError += properties->Metrics->Calculate(testExample->Input(), _neuronNetOutput, visibleStatesCount);
            }
            return sumError / dataSize;
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
