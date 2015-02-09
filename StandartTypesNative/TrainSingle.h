#pragma once

#include "ExportDll.h"

namespace StandardTypesNative {
	class STANDARDTYPES_EXPORT TrainSingle {
	private:
		float *_inputData;
		int _inputLength;
	public:
		TrainSingle(float *inputData, int length);
		virtual ~TrainSingle();
		TrainSingle(const TrainSingle &source);
		float* Input(void) const;
		int InputLength(void) const;
	};
}