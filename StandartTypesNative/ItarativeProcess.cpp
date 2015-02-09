#define STANDARDTYPESAPI
#include "ItarativeProcess.h"

namespace StandardTypesNative {
	void ItarativeProcess::Start(void) {
        if ((ProcessSate == IterativeProcessState::InProgress) || (ProcessSate == IterativeProcessState::Finished)) {
            return;
        }
        if (ProcessSate == IterativeProcessState::NotStarted) {
            FirstRunInit();
        }
        ProcessSate = IterativeProcessState::InProgress;
        RunIterativeProcess();
        if (ProcessSate != IterativeProcessState::Stoped) {
            ProcessSate = IterativeProcessState::Finished;
        }
        ApplyResults();
    }

    void ItarativeProcess::Stop(void) {
        ProcessSate = IterativeProcessState::Stoped;
    }

	void ItarativeProcess::FirstRunInit(void) {
	}

    void ItarativeProcess::ApplyResults(void) {
	}

	void ItarativeProcess::OnIterationCompleted(int iterationNum, float iterationValue, float addedIterationValue) {
		IterationCompleted->Invoke(iterationNum, iterationValue, addedIterationValue);
	}

	void ItarativeProcess::OnIterativeProcessFinished(int iterationCount) {
		IterativeProcessFinished->Invoke(iterationCount);
	}
}