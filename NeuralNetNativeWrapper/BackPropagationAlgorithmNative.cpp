#include "BackPropagationAlgorithmNative.h"
#include "HalfSquaredEuclidianDistance.h"
#include "CrossEntropy.h"
#include "Loglikelihood.h"
#include "L1Regularization.h"
#include "L2Regularization.h"
#include "EliminationRegularization.h"
#include "NoRegularization.h"
#include "MultyLayerPerceptronFactory.h"
#include "HyperbolicTangensFunction.h"
#include "SigmoidFunction.h"
#include "SoftmaxFunction.h"
#include "Callback.h"
#include "malloc.h"
#include "ConstantFactor.h"
#include "ReverseFactor.h"
#include "SqrtReverseFactor.h"

namespace NeuralNetNativeWrapper {
	namespace MultyLayerPerceptronNativeWrapper {
		BackPropagationAlgorithmNative::BackPropagationAlgorithmNative(IList<TrainPair^>^ trainData) {
			AllocateNativeTrainData(trainData);
			_nativeAlgorithm = new NeuralNetNative::MultyLayerPerceptron::BackPropagationAlgorithm(_nativeTrainData, _nativeTrainDataSize);

			_nativeAlgorithm->IterationCompleted = new TripleCallback(gcnew IterationCompletedCallback(this, &BackPropagationAlgorithmNative::IterationCompletedHandler));
			_nativeAlgorithm->IterativeProcessFinished = new SingleCallback(gcnew IterativeProcessFinishedCallback(this, &BackPropagationAlgorithmNative::IterativeProcessFinishedHandler));
		}

		BackPropagationAlgorithmNative::BackPropagationAlgorithmNative(IList<TrainPair^>^ trainData, IList<TrainPair^>^ testData) {
			AllocateNativeTrainData(trainData);
			AllocateNativeTestData(testData);
			_nativeAlgorithm = new NeuralNetNative::MultyLayerPerceptron::BackPropagationAlgorithm(_nativeTrainData, _nativeTrainDataSize, _nativeTestData, _nativeTestDataSize);

			_nativeAlgorithm->IterationCompleted = new TripleCallback(gcnew IterationCompletedCallback(this, &BackPropagationAlgorithmNative::IterationCompletedHandler));
			_nativeAlgorithm->IterativeProcessFinished = new SingleCallback(gcnew IterativeProcessFinishedCallback(this, &BackPropagationAlgorithmNative::IterativeProcessFinishedHandler));
		}
		
		BackPropagationAlgorithmNative::~BackPropagationAlgorithmNative(void) {
			DeleteNativeTrainData();
			DeleteNativeTestData();
			DeleteNativeNeuralNet();
			DeleteNativeProperties();
			DeleteNativeAlgorithm();
		}

		void BackPropagationAlgorithmNative::Start(void) {
			if (_nativeAlgorithm != 0) {
				_nativeAlgorithm->Start();
				ApplyResult();
				DeleteNativeNeuralNet();
				DeleteNativeProperties();
			}
		}

		void BackPropagationAlgorithmNative::Stop(void) {
			if (_nativeAlgorithm != 0) {
				_nativeAlgorithm->Stop();
			}
		}

		void BackPropagationAlgorithmNative::ApplyResult(void) {
			array<BaseNeuralBlock^>^ layers = _multyLayerPerceptron->Layers;
			NeuralNetNative::BaseNeuralBlock** nativeLayers = _nativeNeuralNet->GetLayers();
			int layersCount = layers->Length;
			for (int i = 0; i < layersCount; i++) {
				array<float>^ weights = layers[i]->GetWeights()[0];
				float *nativeWeights = nativeLayers[i]->GetWeights();
				for (int j = 0; j < weights->Length; j++) {
					weights[j] = nativeWeights[j];
				}

				array<float>^ bias = layers[i]->GetBias();
				float *nativeBias = nativeLayers[i]->GetBias();
				for (int j = 0; j < bias->Length; j++) {
					bias[j] = nativeBias[j];
				}
			}
		}

		void BackPropagationAlgorithmNative::CreateNativeNeuralNet(INeuralNet^ neuralNet) {
			_multyLayerPerceptron = dynamic_cast<MultyLayerPerceptron::MultyLayerPerceptron^>(neuralNet);
			array<int>^ layersStruct = _multyLayerPerceptron->GetLayersStruct();
			int inputSize = layersStruct[0];
			int layersCount = layersStruct->Length - 1;
			_nativeLayersStruct = new int[layersCount];
			for (int i = 0; i < layersCount; i++) {
				_nativeLayersStruct[i] = layersStruct[i + 1];
			}

			array<BaseNeuralBlock^>^ layers = _multyLayerPerceptron->Layers;
			IActivationFunction^ hiddenFunction = layers[0]->GetActivationFunction();
			if (dynamic_cast<HyperbolicTangensFunction^>(hiddenFunction) != nullptr) {
				HyperbolicTangensFunction^ hFun = dynamic_cast<HyperbolicTangensFunction^>(hiddenFunction);
				_hiddenActivationFunction = new NeuralNetNative::HyperbolicTangensFunction(hFun->Alpha, hFun->Betta);
			}
			else if (dynamic_cast<SigmoidFunction^>(hiddenFunction) != nullptr) {
				SigmoidFunction^ sFun = dynamic_cast<SigmoidFunction^>(hiddenFunction);
				_hiddenActivationFunction = new NeuralNetNative::SigmoidFunction(sFun->Alpha);
			}
			else {
				_hiddenActivationFunction = new NeuralNetNative::SoftmaxFunction();
			}

			IActivationFunction^ outputFunction = layers[layersCount - 1]->GetActivationFunction();
			if (dynamic_cast<HyperbolicTangensFunction^>(outputFunction) != nullptr) {
				HyperbolicTangensFunction^ hFun = dynamic_cast<HyperbolicTangensFunction^>(outputFunction);
				_outputActivationFunction = new NeuralNetNative::HyperbolicTangensFunction(hFun->Alpha, hFun->Betta);
			}
			else if (dynamic_cast<SigmoidFunction^>(outputFunction) != nullptr) {
				SigmoidFunction^ sFun = dynamic_cast<SigmoidFunction^>(outputFunction);
				_outputActivationFunction = new NeuralNetNative::SigmoidFunction(sFun->Alpha);
			}
			else {
				_outputActivationFunction = new NeuralNetNative::SoftmaxFunction();
			}

			NeuralNetNative::MultyLayerPerceptron::MultyLayerPerceptronFactory *nativeFactory = 
				new NeuralNetNative::MultyLayerPerceptron::MultyLayerPerceptronFactory(inputSize, layersCount, _nativeLayersStruct,
				_hiddenActivationFunction, _outputActivationFunction, NeuralNetNative::StartWeightGenerator::NullDistribution);
			_nativeNeuralNet = (NeuralNetNative::MultyLayerPerceptron::MultyLayerPerceptron*)(nativeFactory->CreateNeuralNet());
			delete nativeFactory;

			NeuralNetNative::BaseNeuralBlock** nativeLayers = _nativeNeuralNet->GetLayers();
			for (int i = 0; i < layersCount; i++) {
				array<float>^ weights = layers[i]->GetWeights()[0];
				float *nativeWeights = nativeLayers[i]->GetWeights();
				for (int j = 0; j < weights->Length; j++) {
					nativeWeights[j] = weights[j];
				}

				array<float>^ bias = layers[i]->GetBias();
				float *nativeBias = nativeLayers[i]->GetBias();
				for (int j = 0; j < bias->Length; j++) {
					nativeBias[j] = bias[j];
				}
			}
		}

		void BackPropagationAlgorithmNative::InitilazeNativeAlgorithm() {
			_nativeAlgorithm->InitilazeMethod(_nativeNeuralNet, _nativeTrainProperties);
		}

		void BackPropagationAlgorithmNative::AllocateNativeTrainData(IList<TrainPair^>^ trainData) {
			_nativeTrainDataSize = trainData->Count;
			_nativeTrainData = new StandardTypesNative::TrainPair*[_nativeTrainDataSize];
			int inputSize = trainData[0]->InputLength;
			int outputSize = trainData[0]->OutputLength;
			for (int i = 0; i < _nativeTrainDataSize; i++) {
				TrainPair^ data = trainData[i];
				
				array<float>^ input = data->Input;
				float *nativeInput = (float*)_mm_malloc(inputSize*sizeof(float), 32);
				for (int j = 0; j < inputSize; j++) {
					nativeInput[j] = input[j];
				}

				array<float>^ output = data->Output;
				float *nativeOutput = (float*)_mm_malloc(outputSize*sizeof(float), 32);
				for (int j = 0; j < outputSize; j++) {
					nativeOutput[j] = output[j];
				}
				
				_nativeTrainData[i] = new StandardTypesNative::TrainPair(nativeInput, nativeOutput, inputSize, outputSize);
			}
		}
		
		void BackPropagationAlgorithmNative::AllocateNativeTestData(IList<TrainPair^>^ testData) {
			_nativeTestDataSize = testData->Count;
			_nativeTestData = new StandardTypesNative::TrainPair*[_nativeTestDataSize];
			int inputSize = testData[0]->InputLength;
			int outputSize = testData[0]->OutputLength;
			for (int i = 0; i < _nativeTestDataSize; i++) {
				TrainPair^ data = testData[i];
				
				array<float>^ input = data->Input;
				float *nativeInput = (float*)_mm_malloc(inputSize*sizeof(float), 32);
				for (int j = 0; j < inputSize; j++) {
					nativeInput[j] = input[j];
				}

				array<float>^ output = data->Output;
				float *nativeOutput = (float*)_mm_malloc(outputSize*sizeof(float), 32);
				for (int j = 0; j < outputSize; j++) {
					nativeOutput[j] = output[j];
				}
				
				_nativeTestData[i] = new StandardTypesNative::TrainPair(nativeInput, nativeOutput, inputSize, outputSize);
			}
		}

		void BackPropagationAlgorithmNative::DeleteNativeNeuralNet(void) {
			if (_nativeNeuralNet != 0) {
				delete _hiddenActivationFunction;
				delete _outputActivationFunction;
				delete [] _nativeLayersStruct;
				delete _nativeNeuralNet;
				_nativeNeuralNet = 0;
			}
		}

		void BackPropagationAlgorithmNative::DeleteNativeTrainData(void) {
			if (_nativeTrainData != 0) {
				for (int i = 0; i < _nativeTrainDataSize; i++) {
					_mm_free(_nativeTrainData[i]);
				}
				delete [] _nativeTrainData;
				_nativeTrainData = 0;
			}
		}

		void BackPropagationAlgorithmNative::DeleteNativeTestData(void) {
			if (_nativeTestData != 0) {
				for (int i = 0; i < _nativeTestDataSize; i++) {
					_mm_free(_nativeTestData[i]);
				}
				delete [] _nativeTestData;
				_nativeTestData = 0;
			}
		}

		void BackPropagationAlgorithmNative::DeleteNativeAlgorithm(void) {
			if (_nativeAlgorithm != 0) {
				delete _nativeAlgorithm->IterationCompleted;
				delete _nativeAlgorithm->IterativeProcessFinished;
				delete _nativeAlgorithm;
				_nativeAlgorithm = 0;
			}
		}
	}
}