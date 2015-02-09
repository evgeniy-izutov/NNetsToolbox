using StandardTypes;

namespace NeuralNet {
    public abstract class TrainMethod : IterativeProcess, ITrainMethod {
    	public abstract void InitilazeMethod(INeuralNet neuralNet, ITrainProperties trainProperties);
    	public abstract ITrainProperties Properties { get; }
    }
}