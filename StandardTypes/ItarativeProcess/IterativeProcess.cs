using System;

namespace StandardTypes {
	public abstract class IterativeProcess {
        public IterativeProcessState ProcessSate { get; protected set; }
        public event EventHandler<IterationCompletedEventArgs> IterationCompleted;
        public event EventHandler<IterativeProcessFinishedEventArgs> IterativeProcessFinished;

        protected void OnIterationCompleted(IterationCompletedEventArgs e) {
            var handler = IterationCompleted;
            if (handler != null) {
                handler(this, e);
            }
        }

        protected void OnIterativeProcessFinished(IterativeProcessFinishedEventArgs e) {
            var handler = IterativeProcessFinished;
            if (handler != null) {
                handler(this, e);
            }
        }

        public void Start() {
            if ((ProcessSate == IterativeProcessState.InProgress)
                || (ProcessSate == IterativeProcessState.Finished)) {
                throw new ApplicationException("Impossble to start process in state: " + ProcessSate.ToString());
            }
            if (ProcessSate == IterativeProcessState.NotStarted) {
                FirstRunInit();
            }
            ProcessSate = IterativeProcessState.InProgress;
            RunIterativeProcess();
            if (ProcessSate != IterativeProcessState.Stoped) {
                ProcessSate = IterativeProcessState.Finished;
            }
            ApplyResults();
        }

        public virtual void Stop() {
            ProcessSate = IterativeProcessState.Stoped;
        }

        protected abstract void RunIterativeProcess();

        protected virtual void FirstRunInit() {}

        protected virtual void ApplyResults() {}

        public enum IterativeProcessState {
            NotStarted,
            InProgress,
            Stoped,
            Finished
        };
    }
}