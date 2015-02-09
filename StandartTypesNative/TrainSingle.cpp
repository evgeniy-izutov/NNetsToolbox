#define STANDARDTYPESAPI
#include "TrainSingle.h"

namespace StandardTypesNative {
	TrainSingle::~TrainSingle() {
		if (_inputData != 0) {
			delete [] _inputData;
			_inputLength = 0;
		}
	}

	TrainSingle::TrainSingle(float *inputData, int length) {
		_inputData = inputData;
		_inputLength = length;
	}

	TrainSingle::TrainSingle(const TrainSingle &source) {
		_inputLength = source._inputLength;
		float *sourceInputData = source._inputData;
		_inputData = new float[_inputLength];
		for(int i = 0; i < _inputLength; i++) {
			_inputData[i] = sourceInputData[i];
		}
	}

	float* TrainSingle::Input(void) const {
		return _inputData;
	}

	int TrainSingle::InputLength(void) const {
		return _inputLength;
	}
}