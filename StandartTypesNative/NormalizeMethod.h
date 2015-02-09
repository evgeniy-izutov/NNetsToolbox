#pragma once

#include "TrainPair.h"

namespace StandardTypesNative {
	class NormalizeMethod {
	public:
		virtual void CollectStatistics(const TrainPair *data, int length) = 0;
    	virtual void NormalizeSet(TrainPair *data, bool isNormalizeOutput = true) = 0;
        virtual void NormalizeInputVector(float *inputVector) = 0;
        virtual void DenormalizeOutputVector(float *outputVector) = 0;
        virtual int InputVectorSize() = 0;
	};
}