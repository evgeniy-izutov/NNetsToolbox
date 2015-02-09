#define STANDARDTYPESAPI
#include "TrainPair.h"

namespace StandardTypesNative {
	TrainPair::~TrainPair(void) {
		if (_outputData != 0) {
			delete [] _outputData;
			_outputLength = 0;
		}
	}

	TrainPair::TrainPair(float* inputData, float* outputData, int inputLength, int outputLength) : TrainSingle(inputData, inputLength) {
		_outputData = outputData;
		_outputLength = outputLength;
	}

	TrainPair::TrainPair(const TrainPair &source) : TrainSingle(source) {
		_outputLength = source._outputLength;
		float *sourceOutputData = source._outputData;
		_outputData = new float[_outputLength];
		for (int i = 0; i < _outputLength; i++) {
			_outputData[i] = sourceOutputData[i];
		}
	}

	float* TrainPair::Output(void) const {
		return _outputData;
	}

	int TrainPair::OutputLength(void) const {
		return _outputLength;
	}
}