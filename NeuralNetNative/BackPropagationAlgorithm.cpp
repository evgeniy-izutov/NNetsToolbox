#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "BackPropagationAlgorithm.h"
#include "TrainPair.h"
#include "malloc.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include "GrainSizeForParallel.h"

using namespace StandardTypesNative;
using namespace tbb;

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		BackPropagationAlgorithm::BackPropagationAlgorithm(StandardTypesNative::TrainPair **trainData, int trainDataSize) {
			_neuralNet = 0;
			_trainDataIterator = new RandomAccessIterator<TrainPair*>(trainData, trainDataSize);
			_testData = 0;
		}

		BackPropagationAlgorithm::BackPropagationAlgorithm(StandardTypesNative::TrainPair **trainData, int trainDataSize, StandardTypesNative::TrainPair **testData, int testDataSize) {
			_neuralNet = 0;
			_trainDataIterator = new RandomAccessIterator<TrainPair*>(trainData, trainDataSize);
			_testData = testData;
			_testDataSize = testDataSize;
		}

		BackPropagationAlgorithm::~BackPropagationAlgorithm(void) {
			delete _trainDataIterator;
			ClearData();
		}

		void BackPropagationAlgorithm::InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties) {
			_neuralNet = dynamic_cast<MultyLayerPerceptron*>(neuralNet);
			if (_neuralNet == 0) {
				return;
			}
			_properties = trainProperties;
			_layers = _neuralNet->GetLayers();
			_layersCount = _neuralNet->GetLayersCount();
			_inputSize = _neuralNet->GetInputSize();
			_outputSize = _neuralNet->GetOutputSize();
			_packageFactor = 1.0f/_properties->PackageSize;
			_packagesCount = CalculatePackagesCount();
			AllocateMemory();
			ProcessSate = IterativeProcessState::NotStarted;
		}

		int BackPropagationAlgorithm::CalculatePackagesCount() {
			int packagesCount = _trainDataIterator->Size()/_properties->PackageSize;
			if (_trainDataIterator->Size()%_properties->PackageSize != 0) {
				packagesCount++;
			}
			return packagesCount;
		}

		TrainProperties* BackPropagationAlgorithm::Properties(void) const {
			return _properties;
		}

		void BackPropagationAlgorithm::AllocateMemory(void) {
			int maxGradientSize = FindMaxSize();

			_gradients = (float*)_mm_malloc(maxGradientSize*sizeof(float), 32);
			_gradientsIntermediate = (float*)_mm_malloc(maxGradientSize*sizeof(float), 32);

			int outputSize = _layers[_layersCount - 1]->GetSize();
			_neuronNetOutput =  (float*)_mm_malloc(outputSize*sizeof(float), 32);
			_partialDerivaitve = (float*)_mm_malloc(outputSize*sizeof(float), 32);

			_oldDeltaWeights = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_derivativeAverages = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_packageDerivative = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_learnFactors = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			for (int i = 0; i < _layersCount; i++) {
				BaseNeuralBlock *layer = _layers[i];
				int weightsCount = (layer->GetPreviousSize())*(layer->GetSize());

				_oldDeltaWeights[i] = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
				_derivativeAverages[i] = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
				_packageDerivative[i] = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
				_learnFactors[i] = (float*)_mm_malloc(weightsCount*sizeof(float), 32);
				for (int j = 0; j < weightsCount; j++) {
					_oldDeltaWeights[i][j] = 0.0f;
					_derivativeAverages[i][j] = 0.0f;
					_packageDerivative[i][j] = 0.0f;
					_learnFactors[i][j] = 1.0f;
				}
			}

			_oldDeltaWeightsForBias = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_derivativeAveragesForBias = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_packageDerivativeForBias = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			_learnFactorsForBias = (float**)_mm_malloc(_layersCount*sizeof(float*), 32);
			for (int i = 0; i < _layersCount; i++) {
				int layerSize = _layers[i]->GetSize();

				_oldDeltaWeightsForBias[i] = (float*)_mm_malloc(layerSize*sizeof(float), 32);
				_derivativeAveragesForBias[i] = (float*)_mm_malloc(layerSize*sizeof(float), 32);
				_packageDerivativeForBias[i] = (float*)_mm_malloc(layerSize*sizeof(float), 32);
				_learnFactorsForBias[i] = (float*)_mm_malloc(layerSize*sizeof(float), 32);
				
				for (int j = 0; j < layerSize; j++) {
					_oldDeltaWeightsForBias[i][j] = 0.0f;
					_derivativeAveragesForBias[i][j] = 0.0f;
					_packageDerivativeForBias[i][j] = 0.0f;
					_learnFactorsForBias[i][j] = 1.0f;
				}
			}
		}

		int BackPropagationAlgorithm::FindMaxSize(void) {
			int maxGradientSize = 0;
			for (int i = 0; i < _layersCount; i++) {
				int curLayerSize = _layers[i]->GetSize();
				if (curLayerSize > maxGradientSize) {
					maxGradientSize = curLayerSize;
				}
			}
			return maxGradientSize;
		}

		void BackPropagationAlgorithm::RunIterativeProcess() {
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

		void BackPropagationAlgorithm::ApplyResults(void) {
			if (ProcessSate == IterativeProcessState::Finished) {
				ClearData();
			}
		}

		void BackPropagationAlgorithm::ClearData(void) {
			if (_neuralNet != 0) {
				_properties = 0;
				_neuralNet = 0;
				_layers = 0;

				for (int i = 0; i < _layersCount; i++) {
					_mm_free(_oldDeltaWeights[i]);
					_mm_free(_derivativeAverages[i]);
					_mm_free(_packageDerivative[i]);
					_mm_free(_oldDeltaWeightsForBias[i]);
					_mm_free(_derivativeAveragesForBias[i]);
					_mm_free(_packageDerivativeForBias[i]);
					_mm_free(_learnFactors[i]);
					_mm_free(_learnFactorsForBias[i]);
				}
				_mm_free(_oldDeltaWeights);
				_mm_free(_derivativeAverages);
				_mm_free(_packageDerivative);
				_mm_free(_oldDeltaWeightsForBias);
				_mm_free(_derivativeAveragesForBias);
				_mm_free(_packageDerivativeForBias);
				_mm_free(_learnFactors);
				_mm_free(_learnFactorsForBias);
				_mm_free(_gradients);
				_mm_free(_gradientsIntermediate);
				_mm_free(_neuronNetOutput);
				_mm_free(_partialDerivaitve);
			}
		}

		float BackPropagationAlgorithm::TestModel(StandardTypesNative::TrainPair **data, int dataSize) {
			if (data == 0 || dataSize == 0) {
				return std::numeric_limits<float>::quiet_NaN();
			}
			else {
				float sumError = 0.0f;
				for (int i = 0; i < dataSize; i++) {
					TrainPair *trainPair = data[i];
					_neuronNetInput = trainPair->Input();
					_neuralNet->Predict(_neuronNetInput, _neuronNetOutput);
					sumError += _properties->Metrics->Calculate(trainPair->Output(), _neuronNetOutput, _outputSize);
				}
				return sumError/dataSize;
			}
		}

		void BackPropagationAlgorithm::TrainEpoch(void) {
			_trainDataIterator->RefreshRandomAccess();
			for (int i = 0; i < _packagesCount; i++) {
				TrainPackage();
			}
		}

		void BackPropagationAlgorithm::TrainPackage(void) {
			for (int i = 0; i < _properties->PackageSize; i++) {
				TrainPair *trainPair = _trainDataIterator->Next();
				_neuronNetInput = trainPair->Input();
				_neuralNet->Predict(_neuronNetInput, _neuronNetOutput);
				_properties->Metrics->CalculatePartialDerivaitve(trainPair->Output(), _neuronNetOutput, _partialDerivaitve, _outputSize);
				CollectWeightsDelta(_partialDerivaitve);
			}
			ModifyWeightsOfNeuronNet();
		}

		void BackPropagationAlgorithm::CollectWeightsDelta(const float *errrorVector) {
			const int firstLayerNumber = 0;
			int lastHiddenLayerNumber = _layersCount - 2;

			CollectWeightsDeltaOfLayer(lastHiddenLayerNumber + 1, LocalGradientForOutputLayer, errrorVector);
			for (int layerNumber = lastHiddenLayerNumber; layerNumber >= firstLayerNumber; layerNumber--) {
				CollectWeightsDeltaOfLayer(layerNumber, LocalGradientForHiddenLayer, 0);
			}
		}

		void BackPropagationAlgorithm::CollectWeightsDeltaOfLayer(int layerNum, LocalGradient localGradientfunction, const float *partialDerivaitve) {
			BaseNeuralBlock *curLayer = _layers[layerNum];
			int prevLayerSize = curLayer->GetPreviousSize();
			int curLayerSize = curLayer->GetSize();
			int nextLayerSize = (layerNum < (_layersCount - 1)) ? _layers[layerNum + 1]->GetSize() : 0;
			float *prevLayerState = (layerNum > 0) ? _layers[layerNum - 1]->GetState() : _neuronNetInput;
			float *curGradients = _gradientsIntermediate;
			float *nextGradients = _gradients;
			float *nextLayerWeights = (layerNum < (_layersCount - 1)) ? _layers[layerNum + 1]->GetWeights() : 0;
			float *packageDerivative = _packageDerivative[layerNum];
			float *packageDerivativeForBias = _packageDerivativeForBias[layerNum];

			(*localGradientfunction)(curGradients, curLayer->GetActivationFunction(), curLayer->GetState(), partialDerivaitve, nextGradients, nextLayerWeights, curLayerSize, nextLayerSize);
			
			parallel_for( blocked_range<size_t>(0, curLayerSize, BPACollectWeightsGrainSize),
			[=](const blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i < r.end(); i++) {
					float localGradient = curGradients[i];
					for (int j = 0; j < prevLayerSize; j++) {
						int weightIndex = prevLayerSize*i + j;
						packageDerivative[weightIndex] -= localGradient*prevLayerState[j];
					}
					packageDerivativeForBias[i] -= localGradient;
				}
			});
						
			_gradients = curGradients;
			_gradientsIntermediate = nextGradients;
		}

		void BackPropagationAlgorithm::ModifyWeightsOfNeuronNet(void) {
			const int firstLayerNumber = 0;
			for (int layerNum = firstLayerNumber; layerNum < _layersCount; layerNum++) {
				BaseNeuralBlock *curLayer = _layers[layerNum];
				int prevLayerSize = curLayer->GetPreviousSize();
				int curLayerSize = curLayer->GetSize();
				float *curLayerWeights = curLayer->GetWeights();
				float *curLayerBias = curLayer->GetBias();
				float *learnFactors = _learnFactors[layerNum];
				float *packageDerivative = _packageDerivative[layerNum];
				float *derivativeAverages = _derivativeAverages[layerNum];
				float *oldDeltaWeights = _oldDeltaWeights[layerNum];
				float *learnFactorsForBias = _learnFactorsForBias[layerNum];
				float *packageDerivativeForBias = _packageDerivativeForBias[layerNum];
				float *oldDeltaWeightsForBias = _oldDeltaWeightsForBias[layerNum];
				float *derivativeAveragesForBias = _derivativeAveragesForBias[layerNum];
				float curLearnSpeed = _properties->BaseLearnSpeed*_properties->FactorStrategy->GetFactor(_epochNumber);

				parallel_for( blocked_range<size_t>(0, curLayerSize, BPAModifyWeightsGrainSize),
				[=](const blocked_range<size_t>& r)
				{
					for (int i = r.begin(); i < r.end(); i++) {
						for (int j = 0; j < prevLayerSize; j++) {
							int weightIndex = prevLayerSize*i + j;

							float lastDerivativeAverage = derivativeAverages[weightIndex];
							float partialDerivative = _packageFactor*packageDerivative[weightIndex] -
								_properties->Regularization->GetDerivative(curLayerWeights[weightIndex]);
							packageDerivative[weightIndex] = 0.0f;
							learnFactors[weightIndex] = (lastDerivativeAverage*partialDerivative > 0.0f) ?
								fminf(learnFactors[weightIndex] + _properties->SpeedBonus, _properties->SpeedUpBorder): 
								fmaxf(learnFactors[weightIndex]*_properties->SpeedPenalty, _properties->SpeedLowBorder);
							derivativeAverages[weightIndex] = _properties->AverageLearnFactor*partialDerivative + 
								(1.0f - _properties->AverageLearnFactor)*lastDerivativeAverage;

							float oldDelta = _properties->Momentum*oldDeltaWeights[weightIndex];
							float newDelta = curLearnSpeed*learnFactors[weightIndex]*partialDerivative + oldDelta;
							oldDeltaWeights[weightIndex] = newDelta;
							curLayerWeights[weightIndex] += (1.0f + _properties->Momentum)*newDelta;
						}

						float lastDerivativeAverageForBias = derivativeAveragesForBias[i];
						float partialDerivativeForBias = _packageFactor*packageDerivativeForBias[i];
						packageDerivativeForBias[i] = 0.0f;
						learnFactorsForBias[i] = (lastDerivativeAverageForBias*partialDerivativeForBias > 0.0f) ?
							fminf(learnFactorsForBias[i] + _properties->SpeedBonus, _properties->SpeedUpBorder):
							fmaxf(learnFactorsForBias[i]*_properties->SpeedPenalty, _properties->SpeedLowBorder);
						derivativeAveragesForBias[i] = _properties->AverageLearnFactor*partialDerivativeForBias + 
							(1.0f - _properties->AverageLearnFactor)*lastDerivativeAverageForBias;

						float oldDeltaForBias = _properties->Momentum*oldDeltaWeightsForBias[i];
						float newDeltaForBias = curLearnSpeed*learnFactorsForBias[i]*partialDerivativeForBias + oldDeltaForBias;
						oldDeltaWeightsForBias[i] = newDeltaForBias;
						curLayerBias[i] += (1.0f + _properties->Momentum)*newDeltaForBias;
					}
				});
			}
		}

		void BackPropagationAlgorithm::LocalGradientForOutputLayer(float *gradientsOutput, ActivationFunction *function, float *state, const float *errors,
				float *nextLayerGradients, float *nextLayerOldWeights, int curLayerSize, int nextLayerSize) {
			
			function->CalculateFirstDerivative(gradientsOutput, errors, state, curLayerSize);
		}

		void BackPropagationAlgorithm::LocalGradientForHiddenLayer(float *gradientsOutput, ActivationFunction *function, float *state, const float *errors,
				float *nextLayerGradients, float *nextLayerOldWeights, int curLayerSize, int nextLayerSize) {
			for (int neuronNum = 0; neuronNum < curLayerSize; neuronNum++) {
				gradientsOutput[neuronNum] = 0.0f;
			}

			parallel_for( blocked_range<size_t>(0, nextLayerSize, BPALocalGradientGrainSize),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float nextLayerGradient = nextLayerGradients[j];
					for (int neuronNum = 0; neuronNum < curLayerSize; neuronNum++) {
						gradientsOutput[neuronNum] += nextLayerGradient*nextLayerOldWeights[curLayerSize*j + neuronNum];
					}
				}
			});

			function->CalculateFirstDerivative(gradientsOutput, state, curLayerSize);
		}
	}
}