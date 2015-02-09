#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include "MultyLayerPerceptronFactory.h"
#include "BaseNeuralBlock.h"
#include "SimpleNeuronBlock.h"
#include "SoftmaxNeuronBlock.h"
#include "SoftmaxFunction.h"
#include <random>

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		MultyLayerPerceptronFactory::MultyLayerPerceptronFactory(int inputSize, int layersCount, int *layersStruct, ActivationFunction *hiddenActivationFunction, ActivationFunction *outputActivationFunction, StartWeightGenerator startWeightGenerator) {
			_inputSize = inputSize;
			_layersCount = layersCount;
			_layersStruct = layersStruct;
			_hiddenLayersActivationFunction = hiddenActivationFunction;
			_outputLayerActivationFunction = outputActivationFunction;
			_startWeightGenerator = startWeightGenerator;
		}

		MultyLayerPerceptronFactory::~MultyLayerPerceptronFactory(void) {
			_layersCount = 0;
			_inputSize = 0;
			_layersStruct = 0;
			_hiddenLayersActivationFunction = 0;
			_outputLayerActivationFunction = 0;
		}

		NeuralNet* MultyLayerPerceptronFactory::CreateNeuralNet(void) {
			MultyLayerPerceptron* neuralNet = new MultyLayerPerceptron(_layersCount);
			BaseNeuralBlock *outputNeuralBlock;
			if (_layersCount > 1) {
				BaseNeuralBlock* lastNeuralBlock = new SimpleNeuronBlock(_layersStruct[0], _inputSize, _hiddenLayersActivationFunction);
				neuralNet->AddNeuralBlock(lastNeuralBlock, 0);
				for (int i = 1; i < _layersCount - 1; i++) {
					lastNeuralBlock = new SimpleNeuronBlock(_layersStruct[i], lastNeuralBlock, _hiddenLayersActivationFunction);
					neuralNet->AddNeuralBlock(lastNeuralBlock, i);
				}

				if (dynamic_cast<SoftmaxFunction*>(_outputLayerActivationFunction)) {
					outputNeuralBlock = new SoftmaxSimpleNeuronBlock(_layersStruct[_layersCount - 1], lastNeuralBlock, _outputLayerActivationFunction);
				}
				else {
					outputNeuralBlock = new SimpleNeuronBlock(_layersStruct[_layersCount - 1], lastNeuralBlock, _outputLayerActivationFunction); 
				}
			}
			else {
				if (dynamic_cast<SoftmaxFunction*>(_outputLayerActivationFunction)) {
					outputNeuralBlock = new SoftmaxSimpleNeuronBlock(_layersStruct[0], _inputSize, _outputLayerActivationFunction);
				}
				else {
					outputNeuralBlock = new SimpleNeuronBlock(_layersStruct[0], _inputSize, _outputLayerActivationFunction); 
				}
			}
			neuralNet->AddNeuralBlock(outputNeuralBlock, _layersCount - 1);

			SetWeigths(neuralNet, _startWeightGenerator);
			return neuralNet;
		}

		void MultyLayerPerceptronFactory::SetWeigths(MultyLayerPerceptron *neuralNet, StartWeightGenerator startWeightGenerator) {
			if (startWeightGenerator == StartWeightGenerator::NullDistribution) {
				return;
			}
	
			BaseNeuralBlock** layers = neuralNet->GetLayers();
			std::random_device rd;
			std::mt19937 gen(rd());

			if (startWeightGenerator == StartWeightGenerator::UniformDistribution) {
				std::uniform_real_distribution<float> dis(-1, 1);
				for (int layerNum = 0; layerNum < _layersCount; layerNum++) {
					int layerSize = layers[layerNum]->GetSize();
					int previousLayerSize = layers[layerNum]->GetPreviousSize();
					float *weights = layers[layerNum]->GetWeights();
					float factor = invsqrtf(previousLayerSize);
					for (int i = 0; i < layerSize*previousLayerSize; i++) {
						weights[i] = factor*dis(gen);
					}
					float *bias = layers[layerNum]->GetBias();
					for (int i = 0; i < layerSize; i++) {
						bias[i] = factor*dis(gen);
					}
				}
			}
			else {
				std::normal_distribution<float> dis(0, 1);
				for (int layerNum = 0; layerNum < _layersCount; layerNum++) {
					int layerSize = layers[layerNum]->GetSize();
					int previousLayerSize = layers[layerNum]->GetPreviousSize();
					float *weights = layers[layerNum]->GetWeights();
					float factor = invsqrtf(previousLayerSize);
					for (int i = 0; i < layerSize*previousLayerSize; i++) {
						weights[i] = factor*dis(gen);
					}
					float *bias = layers[layerNum]->GetBias();
					for (int i = 0; i < layerSize; i++) {
						bias[i] = factor*dis(gen);
					}
				}
			}
		}
	}
}