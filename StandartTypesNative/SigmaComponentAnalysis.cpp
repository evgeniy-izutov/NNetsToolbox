#define STANDARDTYPESAPI
#include "SigmaComponentAnalysis.h"
#include <cmath>

namespace StandardTypesNative {
	SigmaComponentAnalysis::SigmaComponentAnalysis(InvertibleFunction *normalizationFunction) {
		_normalizationFunction = normalizationFunction;
		_isPossibleNormalize = false;
	}

	SigmaComponentAnalysis::~SigmaComponentAnalysis(void) {
		ClearData();
	}

	void SigmaComponentAnalysis::CollectStatistics(const TrainPair *data, int length) {
		ClearData();
		PrepareData(data, length);
		CalcProbabilisticProperties(data);
		_isPossibleNormalize = true;
	}

	void SigmaComponentAnalysis::NormalizeSet(TrainPair *data, bool isNormalizeOutput) {
		_isNormalizeOutput = isNormalizeOutput;
		if (_isPossibleNormalize) {
			ChangeValues(data);
		}
	}

	void SigmaComponentAnalysis::NormalizeInputVector(float *inputVector) {
		if (!_isPossibleNormalize) {
			return;
		}

		if (_normalizationFunction != 0) {
			for (int i = 0; i < _inputVectorSize; i++) {
				float sigma = _inputSigmaVector[i];
				if (sigma == 0.0f) {
					inputVector[i] = 0.0f;
				}
				else {
					inputVector[i] = _normalizationFunction->Calculate((inputVector[i] - _inputMeanValueVector[i])/sigma);
				}
			}
		}
		else {
			for (int i = 0; i < _inputVectorSize; i++) {
				float sigma = _inputSigmaVector[i];
				if (sigma == 0.0f) {
					inputVector[i] = 0.0f;
				}
				else {
					inputVector[i] = (inputVector[i] - _inputMeanValueVector[i])/sigma;
				}
			}
		}
	}

	void SigmaComponentAnalysis::DenormalizeOutputVector(float *outputVector) {
		if (!_isPossibleNormalize || !_isNormalizeOutput) {
			return;
		}

		if (_normalizationFunction != 0) {
			for (int i = 0; i < _outputVectorSize; i++) {
				float meanValue = _outputMeanValueVector[i];
				float sigma = _outputSigmaVector[i];
				outputVector[i] = _normalizationFunction->CalculateInvers(outputVector[i])*sigma + meanValue;
			}
		}
		else {
			for (int i = 0; i < _outputVectorSize; i++) {
				float meanValue = _outputMeanValueVector[i];
				float sigma = _outputSigmaVector[i];
				outputVector[i] = outputVector[i]*sigma + meanValue;
			}
		}
	}

	int SigmaComponentAnalysis::InputVectorSize(void) {
		return _inputVectorSize;
	}

	void SigmaComponentAnalysis::PrepareData(const TrainPair *trainingPairs, int trainingPairsCount) {
		_trainingDataSize = trainingPairsCount;
			
		_inputVectorSize = trainingPairs[0].InputLength();
		_inputMeanValueVector = new float[_inputVectorSize];
		_inputSigmaVector = new float[_inputVectorSize];

		_outputVectorSize = trainingPairs[0].OutputLength();
		_outputMeanValueVector = new float[_outputVectorSize];
		_outputSigmaVector = new float[_outputVectorSize];
	}

	void SigmaComponentAnalysis::CalcProbabilisticProperties(const TrainPair *trainingPairs) {
		for (int i = 0; i < _trainingDataSize; i++) {
			float *inputVector = trainingPairs[i].Input();
			for (int j = 0; j < _inputVectorSize; j++) {
				float value = inputVector[j];
				_inputMeanValueVector[j] += value;
				_inputSigmaVector[j] += value*value;
			}

			float *outputVector = trainingPairs[i].Output();
			for (int j = 0; j < _outputVectorSize; j++) {
				float value = outputVector[j];
				_outputMeanValueVector[j] += value;
				_outputSigmaVector[j] += value*value;
			}
		}

		float meanValueFactor = 1.0f/_trainingDataSize;
		float sigmaFactor = _trainingDataSize/(_trainingDataSize - 1.0f);
		for (int i = 0; i < _inputVectorSize; i++) {
			_inputMeanValueVector[i] *= meanValueFactor;
			float meanValue = _inputMeanValueVector[i];
			float sumSqrValues = _inputSigmaVector[i];
			_inputSigmaVector[i] = sqrt(sigmaFactor*(meanValueFactor*sumSqrValues - meanValue*meanValue));
		}
		for (int i = 0; i < _outputVectorSize; i++) {
			_outputMeanValueVector[i] *= meanValueFactor;
			float meanValue = _outputMeanValueVector[i];
			float sumSqrValues = _outputSigmaVector[i];
			_outputSigmaVector[i] = sqrt(sigmaFactor*(meanValueFactor*sumSqrValues - meanValue*meanValue));
		}
	}

	void SigmaComponentAnalysis::ChangeValues(TrainPair *trainingPairs) {
		for (int i = 0; i < _trainingDataSize; i++) {
			DirectConversationVector(trainingPairs[i].Input(), _inputMeanValueVector, _inputSigmaVector, _inputVectorSize);
			if (_isNormalizeOutput) {
				DirectConversationVector(trainingPairs[i].Output(), _outputMeanValueVector, _outputSigmaVector, _outputVectorSize);
			}
		}
	}

	void SigmaComponentAnalysis::DirectConversationVector(float *vector, const float *meanValueVector, const float *sigmaValueVector, int length) {
		if (_normalizationFunction != 0) {
			for (int i = 0; i < length; i++) {
				float sigma = sigmaValueVector[i];
				if (sigma == 0.0f) {
					vector[i] = 0.0f;
				}
				else {
					vector[i] = _normalizationFunction->Calculate((vector[i] - meanValueVector[i])/sigma);
				}
			}
		}
		else {
			for (int i = 0; i < length; i++) {
				float sigma = sigmaValueVector[i];
				if (sigma == 0.0f) {
					vector[i] = 0.0f;
				}
				else {
					vector[i] = (vector[i] - meanValueVector[i])/sigma;
				}
			}
		}
	}

	void SigmaComponentAnalysis::ClearData(void) {
		if (_inputMeanValueVector != 0) {
			delete [] _inputMeanValueVector;
			delete [] _inputSigmaVector;
			_inputVectorSize = 0;
		}
		if (_outputMeanValueVector != 0) {
			delete [] _outputMeanValueVector;
			delete [] _outputSigmaVector;
			_outputVectorSize = 0;
		}
		_trainingDataSize = 0;
		_isPossibleNormalize = false;
	}
}