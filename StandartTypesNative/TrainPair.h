#pragma once

#include "ExportDll.h"
#include "TrainSingle.h"

namespace StandardTypesNative {
	class STANDARDTYPES_EXPORT TrainPair : public TrainSingle {
	private:
		float* _outputData;
		int _outputLength;
	public:
		TrainPair(float* inputData, float* outputData, int inputLength, int outputLength);
		virtual ~TrainPair(void);
		TrainPair(const TrainPair &source);
		float* Output(void) const;
		int OutputLength(void) const;
	};
}