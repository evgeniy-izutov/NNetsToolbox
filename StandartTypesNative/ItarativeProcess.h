#pragma once

#include "ExportDll.h"
#include "ICallback.h"

namespace StandardTypesNative {
	enum IterativeProcessState {
        NotStarted,
        InProgress,
        Stoped,
        Finished
    };

	class STANDARDTYPES_EXPORT ItarativeProcess {
	protected:
		IterativeProcessState ProcessSate;
	public:
        void Start(void);
        virtual void Stop(void);
		ITripleCallback *IterationCompleted;
		ISingleCallback *IterativeProcessFinished;
	protected:
		virtual void RunIterativeProcess(void) = 0;
        virtual void FirstRunInit(void);
        virtual void ApplyResults(void);
		void OnIterationCompleted(int iterationNum, float iterationValue, float addedIterationValue);
		void OnIterativeProcessFinished(int iterationCount);
	};
}