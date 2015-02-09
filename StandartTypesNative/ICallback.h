#pragma once

class ISingleCallback {
public:
	virtual void Invoke(int iterationCount) = 0;
};

class ITripleCallback {
public:
	virtual void Invoke(int iterationNum, float iterationValue, float addedIterationValue) = 0;
};