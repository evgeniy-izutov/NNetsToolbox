#pragma once

#include "ExportDll.h"
#include "NeuralNet.h"
#include "BaseNeuralBlock.h"

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		class NEURALNETNATIVE_EXPORT MultyLayerPerceptron : public NeuralNet {
		private:
			static const int FirstLayerNum = 0;
			static const int SecondLayerNum = 1;
			int _inputSize;
			int _layersCount;
			BaseNeuralBlock **_layers;
			int _lastLayerNum;
		public:
			MultyLayerPerceptron(void);
			MultyLayerPerceptron(int layersCount);
			~MultyLayerPerceptron(void);
			void AddNeuralBlock(BaseNeuralBlock *block, int layerNum);
			virtual void Predict(const float *input, float *output);
			BaseNeuralBlock** GetLayers(void);
			int GetLayersCount(void);
			int GetInputSize(void);
			int GetOutputSize(void);
		private:
			void CalculateFirstLayer(const float *input);
			void CalculateLeftoverLayers(void);
			void SetOutput(float *output);
		};
	}
}