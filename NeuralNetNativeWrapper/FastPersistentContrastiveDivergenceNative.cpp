#include "FastPersistentContrastiveDivergenceNative.h"
#include "HalfSquaredEuclidianDistance.h"
#include "CrossEntropy.h"
#include "Loglikelihood.h"
#include "HammingDistance.h"
#include "L1Regularization.h"
#include "L2Regularization.h"
#include "NoRegularization.h"
#include "RestrictedBoltzmannMachineFactory.h"
#include "Callback.h"
#include "malloc.h"
#include "ConstantFactor.h"
#include "ReverseFactor.h"
#include "SqrtReverseFactor.h"
#include "LinearGradient.h"
#include "CenteredGradient.h"

namespace NeuralNetNativeWrapper {
	namespace RestrictedBoltzmannMachineNativeWrapper {
		FastPersistentContrastiveDivergenceNative
            ::FastPersistentContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                                        RestrictedBoltzmannMachine::IGradientFunction^ gradient,
                                                        float fastWeightsDecreaseFactor) {
			AllocateNativeTrainData(trainData);

            AllocateNativeGradientFunction(gradient);

			_nativeAlgorithm = new NeuralNetNative::RestrictedBoltzmannMachine
                                   ::FastPersistentContrastiveDivergence(_nativeTrainData, _nativeTrainDataSize,
                                                                         _nativeGradientFunction,
                                                                         fastWeightsDecreaseFactor);

			_nativeAlgorithm->IterationCompleted = new TripleCallback(gcnew IterationCompletedCallback(this, &FastPersistentContrastiveDivergenceNative::IterationCompletedHandler));
			_nativeAlgorithm->IterativeProcessFinished = new SingleCallback(gcnew IterativeProcessFinishedCallback(this, &FastPersistentContrastiveDivergenceNative::IterativeProcessFinishedHandler));
		}

		FastPersistentContrastiveDivergenceNative
            ::FastPersistentContrastiveDivergenceNative(IList<TrainSingle^>^ trainData,
                                                        IList<TrainSingle^>^ testData,
                                                        RestrictedBoltzmannMachine::IGradientFunction^ gradient,
                                                        float fastWeightsDecreaseFactor) {
            AllocateNativeTrainData(trainData);
			AllocateNativeTestData(testData);

            AllocateNativeGradientFunction(gradient);

			_nativeAlgorithm = new NeuralNetNative::RestrictedBoltzmannMachine
                                   ::FastPersistentContrastiveDivergence(_nativeTrainData, _nativeTestData,
                                                                         _nativeTrainDataSize, _nativeTestDataSize,
                                                                         _nativeGradientFunction,
                                                                         fastWeightsDecreaseFactor);

			_nativeAlgorithm->IterationCompleted = new TripleCallback(gcnew IterationCompletedCallback(this, &FastPersistentContrastiveDivergenceNative::IterationCompletedHandler));
			_nativeAlgorithm->IterativeProcessFinished = new SingleCallback(gcnew IterativeProcessFinishedCallback(this, &FastPersistentContrastiveDivergenceNative::IterativeProcessFinishedHandler));
		}
		
		FastPersistentContrastiveDivergenceNative::~FastPersistentContrastiveDivergenceNative(void) {
			DeleteNativeTrainData();
            DeleteNativeTestData();
			DeleteNativeNeuralNet();
			DeleteNativeProperties();
			DeleteNativeAlgorithm();
		}

		void FastPersistentContrastiveDivergenceNative::Start(void) {
			if (_nativeAlgorithm != 0) {
				_nativeAlgorithm->Start();
				ApplyResult();
				DeleteNativeNeuralNet();
				DeleteNativeProperties();
			}
		}

		void FastPersistentContrastiveDivergenceNative::Stop(void) {
			if (_nativeAlgorithm != 0) {
				_nativeAlgorithm->Stop();
			}
		}

		void FastPersistentContrastiveDivergenceNative::ApplyResult(void) {
			array<float>^ weights = _restrictedBoltzmannMachine->Weights;
			float *nativeWeights = _nativeNeuralNet->GetWeights();
			for (int j = 0; j < weights->Length; j++) {
				weights[j] = nativeWeights[j];
			}

			array<float>^ visibleStatesBias = _restrictedBoltzmannMachine->VisibleStatesBias;
			float *nativeVisibleStatesBias = _nativeNeuralNet->GetVisibleStatesBias();
			for (int j = 0; j < visibleStatesBias->Length; j++) {
				visibleStatesBias[j] = nativeVisibleStatesBias[j];
			}

			array<float>^ hiddenStatesBias = _restrictedBoltzmannMachine->HiddenStatesBias;
			float *nativeHiddenStatesBias = _nativeNeuralNet->GetHiddenStatesBias();
			for (int j = 0; j < hiddenStatesBias->Length; j++) {
				hiddenStatesBias[j] = nativeHiddenStatesBias[j];
			}
		}

		void FastPersistentContrastiveDivergenceNative::CreateNativeNeuralNet(INeuralNet^ neuralNet) {
			NeuralNetNative::RestrictedBoltzmannMachine::RbmType rbmType;
			if (dynamic_cast<RestrictedBoltzmannMachine::BinaryBinaryRbm^>(neuralNet) != nullptr) {
				rbmType = NeuralNetNative::RestrictedBoltzmannMachine::BinaryBinary;
			}
			else if (dynamic_cast<RestrictedBoltzmannMachine::BinaryNreluRbm^>(neuralNet) != nullptr) {
				rbmType = NeuralNetNative::RestrictedBoltzmannMachine::BinaryNrelu;
			}
			else if (dynamic_cast<RestrictedBoltzmannMachine::GaussianBinaryRbm^>(neuralNet) != nullptr) {
				rbmType = NeuralNetNative::RestrictedBoltzmannMachine::GaussianBinary;
			}
			else if (dynamic_cast<RestrictedBoltzmannMachine::GaussianNreluRbm^>(neuralNet) != nullptr) {
				rbmType = NeuralNetNative::RestrictedBoltzmannMachine::GaussianNrelu;
			}
			else {
				rbmType = NeuralNetNative::RestrictedBoltzmannMachine::ReluNrelu;
			}

			_restrictedBoltzmannMachine = dynamic_cast<RestrictedBoltzmannMachine::RestrictedBoltzmannMachine^>(neuralNet);

			int visibleStatesCount = _restrictedBoltzmannMachine->VisibleStates->Length;
			int hiddenStatesCount = _restrictedBoltzmannMachine->HiddenStates->Length;
			NeuralNetNative::RestrictedBoltzmannMachine::RestrictedBoltzmannMachineFactory *nativeFactory = 
				new NeuralNetNative::RestrictedBoltzmannMachine::RestrictedBoltzmannMachineFactory(rbmType, visibleStatesCount, hiddenStatesCount, NeuralNetNative::StartWeightGenerator::NullDistribution);
			_nativeNeuralNet = (NeuralNetNative::RestrictedBoltzmannMachine::RestrictedBoltzmannMachineBase*)(nativeFactory->CreateNeuralNet());
			delete nativeFactory;

			array<float>^ weights = _restrictedBoltzmannMachine->Weights;
			float *nativeWeights = _nativeNeuralNet->GetWeights();
			for (int j = 0; j < weights->Length; j++) {
				nativeWeights[j] = weights[j];
			}

			array<float>^ visibleStatesBias = _restrictedBoltzmannMachine->VisibleStatesBias;
			float *nativeVisibleStatesBias = _nativeNeuralNet->GetVisibleStatesBias();
			for (int j = 0; j < visibleStatesBias->Length; j++) {
				nativeVisibleStatesBias[j] = visibleStatesBias[j];
			}

			array<float>^ hiddenStatesBias = _restrictedBoltzmannMachine->HiddenStatesBias;
			float *nativeHiddenStatesBias = _nativeNeuralNet->GetHiddenStatesBias();
			for (int j = 0; j < hiddenStatesBias->Length; j++) {
				nativeHiddenStatesBias[j] = hiddenStatesBias[j];
			}
		}
		
		void FastPersistentContrastiveDivergenceNative::InitilazeNativeAlgorithm() {
			_nativeAlgorithm->InitilazeMethod(_nativeNeuralNet, _nativeTrainProperties);
		}

		void FastPersistentContrastiveDivergenceNative::AllocateNativeTrainData(IList<TrainSingle^>^ trainData) {
			_nativeTrainDataSize = trainData->Count;
			_nativeTrainData = new StandardTypesNative::TrainSingle*[_nativeTrainDataSize];
			int inputSize = trainData[0]->InputLength;
			for (int i = 0; i < _nativeTrainDataSize; i++) {
				TrainSingle^ data = trainData[i];
				
				array<float>^ input = data->Input;
				float *nativeInput = (float*)_mm_malloc(inputSize*sizeof(float), 32);
				for (int j = 0; j < inputSize; j++) {
					nativeInput[j] = input[j];
				}
			
				_nativeTrainData[i] = new StandardTypesNative::TrainSingle(nativeInput, inputSize);
			}
		}
		
		void FastPersistentContrastiveDivergenceNative::AllocateNativeTestData(IList<TrainSingle^>^ testData) {
			_nativeTestDataSize = testData->Count;
			_nativeTestData = new StandardTypesNative::TrainSingle*[_nativeTestDataSize];
			int inputSize = testData[0]->InputLength;
			for (int i = 0; i < _nativeTestDataSize; i++) {
				TrainSingle^ data = testData[i];
				
				array<float>^ input = data->Input;
				float *nativeInput = (float*)_mm_malloc(inputSize*sizeof(float), 32);
				for (int j = 0; j < inputSize; j++) {
					nativeInput[j] = input[j];
				}
				
				_nativeTestData[i] = new StandardTypesNative::TrainSingle(nativeInput, inputSize);
			}
		}

		void FastPersistentContrastiveDivergenceNative::DeleteNativeNeuralNet(void) {
			if (_nativeNeuralNet != 0) {
				delete _nativeNeuralNet;
				_nativeNeuralNet = 0;
			}
		}

		void FastPersistentContrastiveDivergenceNative::DeleteNativeTrainData(void) {
			if (_nativeTrainData != 0) {
				for (int i = 0; i < _nativeTrainDataSize; i++) {
					_mm_free(_nativeTrainData[i]);
				}
				delete [] _nativeTrainData;
				_nativeTrainData = 0;
			}
		}

        void FastPersistentContrastiveDivergenceNative::DeleteNativeTestData(void) {
			if (_nativeTestData != 0) {
				for (int i = 0; i < _nativeTestDataSize; i++) {
					_mm_free(_nativeTestData[i]);
				}
				delete [] _nativeTestData;
				_nativeTestData = 0;
			}
		}

		void FastPersistentContrastiveDivergenceNative::DeleteNativeAlgorithm(void) {
			if (_nativeAlgorithm != 0) {
				delete _nativeAlgorithm->IterationCompleted;
				delete _nativeAlgorithm->IterativeProcessFinished;
				delete _nativeAlgorithm;
				_nativeAlgorithm = 0;
			}
		}

        void FastPersistentContrastiveDivergenceNative::AllocateNativeGradientFunction(RestrictedBoltzmannMachine::IGradientFunction^ gradient) {
		    if (dynamic_cast<RestrictedBoltzmannMachine::LinearGradient^>(gradient) != nullptr) {
		    	_nativeGradientFunction = new NeuralNetNative::RestrictedBoltzmannMachine::LinearGradient();
		    }
            else {
                _nativeGradientFunction = new NeuralNetNative::RestrictedBoltzmannMachine::CenteredGradient();
		    }
        }

        void FastPersistentContrastiveDivergenceNative::DeleteNativeGradientFunction(void) {
            if (_nativeGradientFunction != 0) {
                delete _nativeGradientFunction;
                _nativeGradientFunction = 0;
            }
        }
	}
}