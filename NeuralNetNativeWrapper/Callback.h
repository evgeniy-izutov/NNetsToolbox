#pragma once

#include "ICallback.h"
#include <msclr/auto_gcroot.h>

#pragma managed

namespace NeuralNetNativeWrapper {
	delegate void IterationCompletedCallback(int iterationNum, float iterationValue, float addedIterationValue);
	delegate void IterativeProcessFinishedCallback(int iterationCount);
	
	class SingleCallback : public ISingleCallback {
		gcroot<IterativeProcessFinishedCallback^> m_Managed;
		virtual void Invoke(int iterationCount);
	public:
		SingleCallback(IterativeProcessFinishedCallback^ p_Managed);
	};

	class TripleCallback : public ITripleCallback {
		gcroot<IterationCompletedCallback^> m_Managed;
		virtual void Invoke(int iterationNum, float iterationValue, float addedIterationValue);
	public:
		TripleCallback(IterationCompletedCallback^ p_Managed);
	};
}