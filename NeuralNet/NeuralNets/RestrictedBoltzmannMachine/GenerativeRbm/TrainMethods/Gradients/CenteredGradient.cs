using System;
using System.Linq;

namespace NeuralNet.GenerativeRbm {
	public class CenteredGradient : GradientFunction {
		private readonly float _slidingFactor;
		private float[] _visibleOffsets;
		private float[] _hiddenOffsets;
		private float[] _visibleOffsetsNew;
		private float[] _hiddenOffsetsNew;
		private float[] _dataVisibleHidden;
		private float[] _dataVisible;
		private float[] _dataHidden;
		private float[] _modelVisibleHidden;
		private float[] _modelVisible;
		private float[] _modelHidden;
		private float _packageFactor = 1f;

		public CenteredGradient(float slidingFactor, float[] visibleOffsets, float[] hiddenOffsets) {
			_slidingFactor = slidingFactor;
			
			_visibleOffsets = new float[visibleOffsets.Length];
			Array.Copy(visibleOffsets, _visibleOffsets, visibleOffsets.Length);

			_hiddenOffsets = new float[hiddenOffsets.Length];
			Array.Copy(hiddenOffsets, _hiddenOffsets, hiddenOffsets.Length);
		}

	    public float SlidingFactor {
	        get { return _slidingFactor; }
	    }

	    public float[] VisibleOffsets {
	        get { return _visibleOffsets; }
	    }

	    public float[] HiddenOffsets {
	        get { return _hiddenOffsets; }
	    }

		public override void PrepareToNextPackage(int nextPackageSize) {
			_packageFactor = 1f/nextPackageSize;
			
			Array.Clear(_visibleOffsetsNew, 0, _visibleOffsetsNew.Length);
			Array.Clear(_hiddenOffsetsNew, 0, _hiddenOffsetsNew.Length);

			Array.Clear(_dataVisibleHidden, 0 , _dataVisibleHidden.Length);
			Array.Clear(_modelVisibleHidden, 0, _modelVisibleHidden.Length);

			Array.Clear(_dataHidden, 0, _dataHidden.Length);
			Array.Clear(_modelHidden, 0, _modelHidden.Length);

			Array.Clear(_dataVisible, 0, _dataVisible.Length);
			Array.Clear(_modelVisible, 0, _modelVisible.Length);
		}

		public override void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var hiddenState = hiddenStates[j];
			    var shiftedHiddenState = hiddenState - _hiddenOffsets[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_dataVisibleHidden[j*VisibleStatesCount + i] += (visibleStates[i] - _visibleOffsets[i])*shiftedHiddenState;
				}
				_dataHidden[j] += hiddenState;
				_hiddenOffsetsNew[j] += _packageFactor*hiddenStates[j];
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_dataVisible[i] += visibleStates[i];
				_visibleOffsetsNew[i] += _packageFactor*visibleStates[i];
			}
		}

		public override void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var hiddenState = hiddenStates[j];
                var shiftedHiddenState = hiddenState - _hiddenOffsets[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_modelVisibleHidden[j*VisibleStatesCount + i] += (visibleStates[i] - _visibleOffsets[i])*shiftedHiddenState;
				}
				_modelHidden[j] += hiddenState;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_modelVisible[i] += visibleStates[i];
			}
		}

		public override void MakeGradient(float packageFactor) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var hiddenStateSumGradient = 0f;
				for (var i = 0; i < VisibleStatesCount; i++) {
					var index = j*VisibleStatesCount + i;
					var weightGradient = packageFactor*(_dataVisibleHidden[index] - _modelVisibleHidden[index]);
					hiddenStateSumGradient += _visibleOffsets[i]*weightGradient;
					Gradients.PackageDerivativeForWeights[index] = weightGradient;
				}
				Gradients.PackageDerivativeForHiddenBias[j] = packageFactor*(_dataHidden[j] - _modelHidden[j]) - 
				                                              hiddenStateSumGradient;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				var visibleStateSumGradient = 0f;
				for (var j = 0; j < HiddenStatesCount; j++) {
					visibleStateSumGradient += _hiddenOffsets[j]*Gradients.PackageDerivativeForWeights[j*VisibleStatesCount + i];
				}
				Gradients.PackageDerivativeForVisibleBias[i] = packageFactor*(_dataVisible[i] - _modelVisible[i]) -
				                                               visibleStateSumGradient;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_visibleOffsets[i] = (1f - _slidingFactor)*_visibleOffsets[i] +
				                          _slidingFactor*_visibleOffsetsNew[i];
			}

			for (var j = 0; j < HiddenStatesCount; j++) {
			    _hiddenOffsets[j] = (1f - _slidingFactor)*_hiddenOffsets[j] +
				                          _slidingFactor*_hiddenOffsetsNew[j];
			}
		}

		protected override void AllocateMemory() {
		    _dataVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			_modelVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			
			_dataVisible = new float[VisibleStatesCount];
			_modelVisible = new float[VisibleStatesCount];
			
			_dataHidden = new float[HiddenStatesCount];
			_modelHidden = new float[HiddenStatesCount];

            _visibleOffsets = _visibleOffsets ?? Enumerable.Repeat(0.5f, VisibleStatesCount).ToArray();
			_hiddenOffsets = _hiddenOffsets ?? Enumerable.Repeat(0.5f, HiddenStatesCount).ToArray();

			_visibleOffsetsNew = Enumerable.Repeat(0f, VisibleStatesCount).ToArray();
			_hiddenOffsetsNew = Enumerable.Repeat(0f, HiddenStatesCount).ToArray();
		}
	}
}
