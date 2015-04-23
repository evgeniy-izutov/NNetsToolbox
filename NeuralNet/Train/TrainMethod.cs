using StandardTypes;

namespace NeuralNet {
    public abstract class TrainMethod<T> : IterativeProcess, ITrainMethod<T> where T:TrainData {
    	public abstract void InitilazeMethod(INeuralNet neuralNet, ITrainProperties<T> trainProperties);
    	public abstract ITrainProperties<T> Properties { get; }
    }
}
