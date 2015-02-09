#pragma once

#include "Callback.h"

#pragma managed

namespace NeuralNetNativeWrapper {
	void SingleCallback::Invoke(int iterationCount) {
		m_Managed->Invoke(iterationCount);
	}

	SingleCallback::SingleCallback(IterativeProcessFinishedCallback^ p_Managed) : m_Managed(p_Managed) {
	}
	
	void TripleCallback::Invoke(int iterationNum, float iterationValue, float addedIterationValue) {
		m_Managed->Invoke(iterationNum, iterationValue, addedIterationValue);
	}

	TripleCallback::TripleCallback(IterationCompletedCallback^ p_Managed) : m_Managed(p_Managed) {
	}
}