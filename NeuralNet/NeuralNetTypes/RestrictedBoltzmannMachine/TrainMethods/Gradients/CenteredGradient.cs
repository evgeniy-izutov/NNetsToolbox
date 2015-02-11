using System;
using System.Linq;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public class CenteredGradient : GradientFunction {
		private readonly InitializeType _initializeType;
		private readonly float _slidingAverage;
		private readonly float _startVisibleOffsetScalar;
		private readonly float _startHiddenOffsetScalar;
		private float[] _visibleOffsetVector;
		private float[] _hiddenOffsetVector;
		private float[] _visibleOffsetNewVector;
		private float[] _hiddenOffsetNewVector;
		private float[] _dataVisibleHidden;
		private float[] _dataVisible;
		private float[] _dataHidden;
		private float[] _modelVisibleHidden;
		private float[] _modelVisible;
		private float[] _modelHidden;
		private float _packageFactor = 1f;

		private enum InitializeType {
			Constant,
			Vector
		}

		public CenteredGradient(float slidingAverage, float startVisibleOffsetScalar, float startHiddenOffsetScalar = 0.5f) {
			_slidingAverage = slidingAverage;
			_startVisibleOffsetScalar = startVisibleOffsetScalar;
			_startHiddenOffsetScalar = startHiddenOffsetScalar;
			
			_initializeType = InitializeType.Constant;
		}

		public CenteredGradient(float slidingAverage, float[] startVisibleOffsetVector, float[] startHiddenOffsetVector = null) {
			_slidingAverage = slidingAverage;
			
			if (startVisibleOffsetVector != null) {
				_visibleOffsetVector = new float[startVisibleOffsetVector.Length];
				Array.Copy(startVisibleOffsetVector, _visibleOffsetVector, startVisibleOffsetVector.Length);
			}

			if (startHiddenOffsetVector != null) {
				_visibleOffsetVector = new float[startHiddenOffsetVector.Length];
				Array.Copy(startHiddenOffsetVector, _hiddenOffsetVector, startHiddenOffsetVector.Length);
			}
			
			_initializeType = InitializeType.Vector;
		}

		public override void PrepareToNextPackage(int nextPackageSize) {
			_packageFactor = 1f/nextPackageSize;
			
			Array.Clear(_visibleOffsetNewVector, 0, _visibleOffsetNewVector.Length);
			Array.Clear(_hiddenOffsetNewVector, 0, _hiddenOffsetNewVector.Length);

			Array.Clear(_dataVisibleHidden, 0 , _dataVisibleHidden.Length);
			Array.Clear(_modelVisibleHidden, 0, _modelVisibleHidden.Length);

			Array.Clear(_dataHidden, 0, _dataHidden.Length);
			Array.Clear(_modelHidden, 0, _modelHidden.Length);

			Array.Clear(_dataVisible, 0, _dataVisible.Length);
			Array.Clear(_modelVisible, 0, _modelVisible.Length);
		}

		public override void StorePositivePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_dataVisibleHidden[startIndex + i] += (visibleStates[i] - _visibleOffsetVector[i])*(hiddenState - _hiddenOffsetVector[j]);
				}
				_dataHidden[j] += hiddenState;
				_hiddenOffsetNewVector[j] += _packageFactor*hiddenStates[j];
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_dataVisible[i] += visibleStates[i];
				_visibleOffsetNewVector[i] += _packageFactor*visibleStates[i];
			}
		}

		public override void StoreNegativePhaseData(float[] visibleStates, float[] hiddenStates) {
			for (var j = 0; j < HiddenStatesCount; j++) {
				var startIndex = j*VisibleStatesCount;
				var hiddenState = hiddenStates[j];
				for (var i = 0; i < VisibleStatesCount; i++) {
					_modelVisibleHidden[startIndex + i] += visibleStates[i]*hiddenState;
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
					hiddenStateSumGradient += _visibleOffsetVector[i]*weightGradient;
					Gradients.PackageDerivativeForWeights[index] = weightGradient;
				}
				Gradients.PackageDerivativeForHiddenBias[j] = packageFactor*(_dataHidden[j] - _modelHidden[j]) - 
				                                              hiddenStateSumGradient;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				var visibleStateSumGradient = 0f;
				for (var j = 0; j < HiddenStatesCount; j++) {
					visibleStateSumGradient += _hiddenOffsetVector[j]*Gradients.PackageDerivativeForWeights[j*VisibleStatesCount + i];
				}
				Gradients.PackageDerivativeForVisibleBias[i] = packageFactor*(_dataVisible[i] - _modelVisible[i]) -
				                                               visibleStateSumGradient;
			}

			for (var i = 0; i < VisibleStatesCount; i++) {
				_visibleOffsetVector[i] = (1f - _slidingAverage)*_visibleOffsetVector[i] +
				                          _slidingAverage*_visibleOffsetNewVector[i];
			}

			for (var j = 0; j < HiddenStatesCount; j++) {
			    _hiddenOffsetVector[j] = (1f - _slidingAverage)*_hiddenOffsetVector[j] +
				                          _slidingAverage*_hiddenOffsetNewVector[j];
			}
		}

		protected override void AllocateMemory() {
		    _dataVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			_modelVisibleHidden = new float[VisibleStatesCount*HiddenStatesCount];
			
			_dataVisible = new float[VisibleStatesCount];
			_modelVisible = new float[VisibleStatesCount];
			
			_dataHidden = new float[HiddenStatesCount];
			_modelHidden = new float[HiddenStatesCount];

			switch (_initializeType) {
				case InitializeType.Constant:
					_visibleOffsetVector = Enumerable.Repeat(_startVisibleOffsetScalar, VisibleStatesCount).ToArray();
					_hiddenOffsetVector = Enumerable.Repeat(_startHiddenOffsetScalar, HiddenStatesCount).ToArray();
					break;
				case InitializeType.Vector:
					_visibleOffsetVector = _visibleOffsetVector ?? Enumerable.Repeat(0.5f, VisibleStatesCount).ToArray();
					_hiddenOffsetVector = _hiddenOffsetVector ?? Enumerable.Repeat(0.5f, HiddenStatesCount).ToArray();
					break;
			}
			_visibleOffsetNewVector = Enumerable.Repeat(0f, VisibleStatesCount).ToArray();
			_hiddenOffsetNewVector = Enumerable.Repeat(0f, HiddenStatesCount).ToArray();
		}
	}
}
