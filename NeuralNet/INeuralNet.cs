namespace NeuralNet {
    public interface INeuralNet {
        void Predict(float[] input, float[] output);
        byte[] SaveState();
        void LoadState(byte[] state);
    }
}