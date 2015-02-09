#define STANDARDTYPESAPI
#include "MinMaxComponentAnalysis.h"

namespace StandardTypesNative {
	MinMaxComponentAnalysis::MinMaxComponentAnalysis(InvertibleFunction *normalizationFunction) {
		_normalizationFunction = normalizationFunction;
		_isPossibleNormalize = false;
	}

	MinMaxComponentAnalysis::~MinMaxComponentAnalysis() {
		ClearData();
	}

	void MinMaxComponentAnalysis::CollectStatistics(const TrainPair *data, int length) {
		ClearData();
		PrepareData(data, length);
		CalcProbabilisticProperties(data);
		_isPossibleNormalize = true;
	}

	void MinMaxComponentAnalysis::NormalizeSet(TrainPair *data, bool isNormalizeOutput) {
		_isNormalizeOutput = isNormalizeOutput;
		if (_isPossibleNormalize) {
			ChangeValues(data);
		}
	}

	void MinMaxComponentAnalysis::NormalizeInputVector(float *inputVector) {
		if (!_isPossibleNormalize) {
			return;
		}

		if (_normalizationFunction != 0) {
			for (int i = 0; i < _inputVectorSize; i++) {
				float minValue = _inputMinVector[i];
				float maxValue = _inputMaxVector[i];
				float dif = maxValue - minValue;
				if (dif == 0.0f) {
					inputVector[i] = 0.0f;
				}
				else {
					inputVector[i] = _normalizationFunction->Calculate((inputVector[i] - minValue)/dif);
				}
			}
		}
		else {
			for (int i = 0; i < _inputVectorSize; i++) {
				float minValue = _inputMinVector[i];
				float maxValue = _inputMaxVector[i];
				float dif = maxValue - minValue;
				if (dif == 0.0f) {
					inputVector[i] = 0.0f;
				}
				else {
					inputVector[i] = (inputVector[i] - minValue)/dif;
				}
			}
		}
	}

	void MinMaxComponentAnalysis::DenormalizeOutputVector(float *outputVector) {
		if (!_isPossibleNormalize || !_isNormalizeOutput) {
			return;
		}

		if (_normalizationFunction != 0) {
			for (int i = 0; i < _outputVectorSize; i++) {
				float minValue = _outputMinVector[i];
				float maxValue = _outputMaxVector[i];
				float dif = maxValue - minValue;
				float value = outputVector[i];
				outputVector[i] = _normalizationFunction->CalculateInvers(value)*dif + minValue;
			}
		}
		else {
			for (int i = 0; i < _outputVectorSize; i++) {
				float minValue = _outputMinVector[i];
				float maxValue = _outputMaxVector[i];
				float dif = maxValue - minValue;
				float value = outputVector[i];
				outputVector[i] = value*dif + minValue;
			}
		}
	}

	int MinMaxComponentAnalysis::InputVectorSize() {
		return _inputVectorSize;
	}

	void MinMaxComponentAnalysis::PrepareData(const TrainPair *trainingPairs, int trainingPairsCount) {
		_trainingDataSize = trainingPairsCount;
			
		_inputVectorSize = trainingPairs[0].InputLength();
		_inputMinVector = new float[_inputVectorSize];
		_inputMaxVector = new float[_inputVectorSize];

		_outputVectorSize = trainingPairs[0].OutputLength();
		_outputMinVector = new float[_outputVectorSize];
		_outputMaxVector = new float[_outputVectorSize];
	}

	void MinMaxComponentAnalysis::CalcProbabilisticProperties(const TrainPair *trainingPairs) {
		for (int i = 0; i < _trainingDataSize; i++) {
			float *inputVector = trainingPairs[i].Input();
			for (int j = 0; j < _inputVectorSize; j++) {
				float value = inputVector[j];
				if (value < _inputMinVector[j]) {
					_inputMinVector[j] = value;
				}
				else if (value > _inputMaxVector[j]) {
					_inputMaxVector[j] = value;
				}
			}

			float *outputVector = trainingPairs[i].Output();
			for (int j = 0; j < _outputVectorSize; j++) {
				float value = outputVector[j];
				if (value < _outputMinVector[j]) {
					_outputMinVector[j] = value;
				}
				else if (value > _outputMaxVector[j]) {
					_outputMaxVector[j] = value;
				}
			}
		}
	}

	void MinMaxComponentAnalysis::ChangeValues(TrainPair *trainingPairs) {
		for (int i = 0; i < _trainingDataSize; i++) {
			DirectConversationVector(trainingPairs[i].Input(), _inputMinVector, _inputMaxVector, _inputVectorSize);
			if (_isNormalizeOutput) {
				DirectConversationVector(trainingPairs[i].Output(), _outputMinVector, _outputMaxVector, _outputVectorSize);
			}
		}
	}

	void MinMaxComponentAnalysis::DirectConversationVector(float *vector, const float *minVector, const float *maxVector, int length) {
		if (_normalizationFunction != 0) {
			for (int i = 0; i < length; i++) {
				float minValue = minVector[i];
				float maxValue = maxVector[i];
				float dif = maxValue - minValue;
				if (dif == 0.0f) {
					vector[i] = 0.0f;
				}
				else {
					vector[i] = _normalizationFunction->Calculate((vector[i] - minValue)/dif);
				}
			}
		}
		else {
			for (int i = 0; i < length; i++) {
				float minValue = minVector[i];
				float maxValue = maxVector[i];
				float dif = maxValue - minValue;
				if (dif == 0.0f) {
					vector[i] = 0.0f;
				}
				else {
					vector[i] = (vector[i] - minValue)/dif;
				}
			}
		}
	}

	void MinMaxComponentAnalysis::ClearData(void) {
		if (_inputMinVector != 0) {
			delete [] _inputMinVector;
			delete [] _inputMaxVector;
			_inputVectorSize = 0;
		}
		if (_outputMinVector != 0) {
			delete [] _outputMinVector;
			delete [] _outputMaxVector;
			_outputVectorSize = 0;
		}
		_trainingDataSize = 0;
		_isPossibleNormalize = false;
	}
}