namespace NeuralNet {
    public interface INeuralNet {
        void Predict(float[] input, float[] output);
        void Save(string path);
        void Load(string path);
    }
}