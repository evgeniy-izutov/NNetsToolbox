namespace NeuralNet.GenerativeRbm {
	public interface IGradientFunction {
		void Initialize(RbmGradients gradients);
		void PrepareToNextPackage(int nextPackageSize);
		void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates);
		void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates);
		void MakeGradient(float packageFactor);
	}
}
