using System;

namespace NeuralNet.GenerativeRbm {
	public sealed class RbmGradients {
		private readonly float[] _packageDerivativeForWeights;
		private readonly float[] _packageDerivativeForVisibleBias;
		private readonly float[] _packageDerivativeForHiddenBias;

		public RbmGradients(int visibleStatesCount, int hiddenStatesCount) {
			_packageDerivativeForWeights = new float[visibleStatesCount*hiddenStatesCount];
			Array.Clear(_packageDerivativeForWeights, 0 , visibleStatesCount*hiddenStatesCount);
			
			_packageDerivativeForVisibleBias = new float[visibleStatesCount];
			Array.Clear(_packageDerivativeForVisibleBias, 0 , visibleStatesCount);

			_packageDerivativeForHiddenBias = new float[hiddenStatesCount];
			Array.Clear(_packageDerivativeForHiddenBias, 0 , hiddenStatesCount);
		}

		public int VisibleStatesCount {
			get { return _packageDerivativeForVisibleBias.Length; }
		}

		public int HiddenStatesCount {
			get { return _packageDerivativeForHiddenBias.Length; }
		}

		public float[] PackageDerivativeForWeights {
			get { return _packageDerivativeForWeights; }
		}

		public float[] PackageDerivativeForVisibleBias {
			get { return _packageDerivativeForVisibleBias; }
		}

		public float[] PackageDerivativeForHiddenBias {
			get { return _packageDerivativeForHiddenBias; }
		}
	}
}
