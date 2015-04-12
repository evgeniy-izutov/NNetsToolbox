using System;
using StandardTypes;

namespace NeuralNet {
	public interface ITrainMethod {
		void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties);
    	ITrainProperties Properties { get; }
		void Start();
		void Stop();
		event EventHandler<IterationCompletedEventArgs> IterationCompleted;
        event EventHandler<IterativeProcessFinishedEventArgs> IterativeProcessFinished;
	}
}
