namespace NeuralNet.RestrictedBoltzmannMachine {
	public abstract class GradientFunction : IGradientFunction {
		protected RbmGradients Gradients;
		protected int VisibleStatesCount;
		protected int HiddenStatesCount;

		public void Initialize(RbmGradients gradients) {
			Gradients = gradients;

			VisibleStatesCount = gradients.VisibleStatesCount;
			HiddenStatesCount = gradients.HiddenStatesCount;

			AllocateMemory();
		}

		public abstract void PrepareToNextPackage(int nextPackageSize);

		public abstract void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates);

		public abstract void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates);

		public abstract void MakeGradient(float packageFactor);

		protected virtual void AllocateMemory() {}
	}
}
