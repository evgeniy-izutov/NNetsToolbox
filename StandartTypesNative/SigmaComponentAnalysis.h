#pragma once

#include "NormalizeMethod.h"
#include "InvertibleFunction.h"
#include "ExportDll.h"

namespace StandardTypesNative {
	class STANDARDTYPES_EXPORT SigmaComponentAnalysis : public NormalizeMethod {
	private:
		InvertibleFunction *_normalizationFunction;
		float *_inputMeanValueVector;
		float *_inputSigmaVector;
		float *_outputMeanValueVector;
		float *_outputSigmaVector;
		int _inputVectorSize;
		int _outputVectorSize;
		int _trainingDataSize;
		bool _isPossibleNormalize;
		bool _isNormalizeOutput;
	public:
		SigmaComponentAnalysis(InvertibleFunction *normalizationFunction = 0);
		~SigmaComponentAnalysis(void);
		virtual void CollectStatistics(const TrainPair *data, int length);
		virtual void NormalizeSet(TrainPair *data, bool isNormalizeOutput);
		virtual void NormalizeInputVector(float *inputVector);
		virtual void DenormalizeOutputVector(float *outputVector);
		virtual int InputVectorSize(void);
	private:
		void PrepareData(const TrainPair *trainingPairs, int trainingPairsCount);
		void CalcProbabilisticProperties(const TrainPair *trainingPairs);
		void ChangeValues(TrainPair *trainingPairs);
		void DirectConversationVector(float *vector, const float *meanValueVector, const float *sigmaValueVector, int length);
		void ClearData(void);
	};
}