namespace NeuralNet {
    public interface INeuralNet {
        void Predict(float[] input, float[] output);
	    float[] Predict(float[] input);
        byte[] SaveState();
        void LoadState(byte[] state);
    }
}