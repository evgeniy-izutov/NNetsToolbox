#pragma once

#include "ExportDll.h"
#include "NeuralNetFactory.h"
#include "ActivationFunction.h"
#include "MultyLayerPerceptron.h"

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		class NEURALNETNATIVE_EXPORT MultyLayerPerceptronFactory : public NeuralNetFactory {
		private:
			int _inputSize;
			int _layersCount;
			int *_layersStruct;
			ActivationFunction *_hiddenLayersActivationFunction;
			ActivationFunction *_outputLayerActivationFunction;
			StartWeightGenerator _startWeightGenerator;
		public:
			MultyLayerPerceptronFactory(int inputSize, int layersCount, int *layersStruct, ActivationFunction *hiddenActivationFunction, ActivationFunction *outputActivationFunction, StartWeightGenerator startWeightGenerator);
			~MultyLayerPerceptronFactory(void);
			NeuralNet* CreateNeuralNet(void);
		private:
			void SetWeigths(MultyLayerPerceptron *neuralNet, StartWeightGenerator startWeightGenerator);
		};
	}
}