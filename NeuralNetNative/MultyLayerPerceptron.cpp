#define NEURALNETNATIVEAPI
#include "MultyLayerPerceptron.h"

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		MultyLayerPerceptron::MultyLayerPerceptron(void) {
		}

		MultyLayerPerceptron::MultyLayerPerceptron(int layersCount) {
			_layersCount = layersCount;
			_lastLayerNum = layersCount - 1;
			_layers = new BaseNeuralBlock*[layersCount];
		}

		MultyLayerPerceptron::~MultyLayerPerceptron(void) {
			_layersCount = 0;
			if (_layers != 0) {
				for (int i = 0; i < _layersCount; i++) {
					if (_layers[i] != 0) {
						delete _layers[i];
					}
				}
			}
			delete [] _layers;
		}

		void MultyLayerPerceptron::AddNeuralBlock(BaseNeuralBlock *block, int layerNum) {
			if ((layerNum < 0) && (layerNum >= _layersCount)) {
				return;
			}
			_layers[layerNum] = block;
			if (layerNum == FirstLayerNum) {
				_inputSize = block->GetPreviousSize();
			}
		}

		void MultyLayerPerceptron::Predict(const float *input, float *output) {
			CalculateFirstLayer(input);
			CalculateLeftoverLayers();
			SetOutput(output);
		}

		BaseNeuralBlock** MultyLayerPerceptron::GetLayers(void) {
			return _layers;
		}

		int MultyLayerPerceptron::GetLayersCount(void) {
			return _layersCount;
		}
		
		int MultyLayerPerceptron::GetInputSize(void) {
			return _inputSize;
		}

		int MultyLayerPerceptron::GetOutputSize(void) {
			return _layers[_lastLayerNum]->GetSize();
		}

		void MultyLayerPerceptron::CalculateFirstLayer(const float *input) {
			_layers[FirstLayerNum]->Calculate(input);
		}

		void MultyLayerPerceptron::CalculateLeftoverLayers(void) {
			for (int layerNum = SecondLayerNum; layerNum < _layersCount; layerNum++) {
				_layers[layerNum]->Calculate();
			}
		}

		void MultyLayerPerceptron::SetOutput(float *output) {
			float* neuronNetOutput = _layers[_lastLayerNum]->GetState();
			int lastLayerSize = _layers[_lastLayerNum]->GetSize();
			for (int i = 0; i < lastLayerSize; i++) {
				output[i] = neuronNetOutput[i];
			}
		}
	}
}