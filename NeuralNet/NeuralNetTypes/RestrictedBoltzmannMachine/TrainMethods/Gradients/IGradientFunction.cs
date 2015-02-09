namespace NeuralNet.RestrictedBoltzmannMachine {
	public interface IGradientFunction {
		void StorePositivePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates);
		void StoreNegativePhaseData(RbmGradients gradients, float[] visibleStates, float[] hiddenStates);
		void MakeGradient(RbmGradients gradients, float packageFactor);
	}
}