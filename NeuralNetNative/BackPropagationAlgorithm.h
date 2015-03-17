#pragma once

#include "ExportDll.h"
#include "TrainMethod.h"
#include "TrainProperties.h"
#include "RandomAccessIterator.h"
#include "TrainPair.h"
#include "MultyLayerPerceptron.h"
#include "ActivationFunction.h"

namespace NeuralNetNative {
	namespace MultyLayerPerceptron {
		typedef void (*LocalGradient)(float *gradientsOutput, ActivationFunction *function, float *net, const float *errors,
			float *nextLayerGradients, float *nextLayerOldWeights, int curLayerSize, int nextLayerSize);
		
		class NEURALNETNATIVE_EXPORT BackPropagationAlgorithm : public TrainMethod {
		private:
			StandardTypesNative::RandomAccessIterator<StandardTypesNative::TrainPair*> *_trainDataIterator;
			StandardTypesNative::TrainPair **_testData;
			int _testDataSize;
			TrainProperties *_properties;
			MultyLayerPerceptron *_neuralNet;
			BaseNeuralBlock **_layers;
			int _layersCount;
			int _inputSize;
			int _outputSize;
			float *_neuronNetOutput;
			float *_neuronNetInput;
			float *_partialDerivaitve;
			float **_oldDeltaWeights;
			float **_derivativeAverages;
			float **_packageDerivative;
			float **_learnFactors;
			float **_oldDeltaWeightsForBias;
			float ** _derivativeAveragesForBias;
			float **_packageDerivativeForBias;
			float **_learnFactorsForBias;
			float *_gradients;
			float *_gradientsIntermediate;
			float _packageFactor;
			float _epochNumber;
			int _packagesCount;
		public:
			BackPropagationAlgorithm(StandardTypesNative::TrainPair **trainData, int trainDataSize);
			BackPropagationAlgorithm(StandardTypesNative::TrainPair **trainData, int trainDataSize, StandardTypesNative::TrainPair **testData, int testDataSize);
			~BackPropagationAlgorithm(void);
			virtual void InitilazeMethod(NeuralNet *neuralNet, TrainProperties *trainProperties);
			virtual TrainProperties* Properties(void) const;
		private:
			void AllocateMemory(void);
            bool IsTestDataAvailable() const;
            void RunTraingWithTesting(void);
            void RunTraingWithoutTesting(void);
			int CalculatePackagesCount(void);
			int FindMaxSize(void);
			virtual void RunIterativeProcess(void);
			virtual void ApplyResults(void);
			void ClearData(void);
			float TestModel(StandardTypesNative::TrainPair **data, int dataSize);
			void TrainEpoch(void);
			void TrainPackage(void);
			void CollectWeightsDelta(const float *errrorVector);
			void CollectWeightsDeltaOfLayer(int layerNum, LocalGradient localGradientfunction, const float *errorVector);
			void ModifyWeightsOfNeuronNet(void);
			static void LocalGradientForOutputLayer(float *gradientsOutput, ActivationFunction *function, float *net, const float *errors,
				float *nextLayerGradients, float *nextLayerOldWeights, int curLayerSize, int nextLayerSize);
			static void LocalGradientForHiddenLayer(float *gradientsOutput, ActivationFunction *function, float *net, const float *errors,
				float *nextLayerGradients, float *nextLayerOldWeights, int curLayerSize, int nextLayerSize);
		};
	}
}