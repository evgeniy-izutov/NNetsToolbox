#pragma once

#include "NormalizeMethod.h"
#include "InvertibleFunction.h"
#include "ExportDll.h"

namespace StandardTypesNative {
	class STANDARDTYPES_EXPORT MinMaxComponentAnalysis : public NormalizeMethod {
	private:
		InvertibleFunction *_normalizationFunction;
		float *_inputMinVector;
		float *_inputMaxVector;
		float *_outputMinVector;
		float *_outputMaxVector;
		int _inputVectorSize;
		int _outputVectorSize;
		int _trainingDataSize;
		bool _isPossibleNormalize;
		bool _isNormalizeOutput;
	public:
		MinMaxComponentAnalysis(InvertibleFunction *normalizationFunction = 0);
		~MinMaxComponentAnalysis();
		virtual void CollectStatistics(const TrainPair *data, int length);
		virtual void NormalizeSet(TrainPair *data, bool isNormalizeOutput);
		virtual void NormalizeInputVector(float *inputVector);
		virtual void DenormalizeOutputVector(float *outputVector);
		virtual int InputVectorSize();
	private:
		void PrepareData(const TrainPair *trainingPairs, int trainingPairsCount);
		void CalcProbabilisticProperties(const TrainPair *trainingPairs);
		void ChangeValues(TrainPair *trainingPairs);
		void DirectConversationVector(float *vector, const float *minVector, const float *maxVector, int length);
		void ClearData(void);
	};
}