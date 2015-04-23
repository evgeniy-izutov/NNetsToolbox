using System;
using StandardTypes;

namespace NeuralNet {
	public interface ITrainMethod<T> where T:TrainData {
		void InitilazeMethod(INeuralNet neuralNet, ITrainProperties<T> trainProperties);
    	ITrainProperties<T> Properties { get; }
		void Start();
		void Stop();
		event EventHandler<IterationCompletedEventArgs> IterationCompleted;
        event EventHandler<IterativeProcessFinishedEventArgs> IterativeProcessFinished;
	}
}
