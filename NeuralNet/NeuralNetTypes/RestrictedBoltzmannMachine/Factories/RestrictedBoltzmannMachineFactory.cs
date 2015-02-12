using System;
using StandardTypes;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public sealed class RestrictedBoltzmannMachineFactory : INeuralNetFactory {
		private const float BiasStartValueBorder = 20.0f;
		private readonly int _visibleStatesCount;
		private readonly int _hiddenStatesCount;
		private readonly DistributionType _startWeightGenerator;
		private readonly RbmType _rbmType;
		private readonly float[] _inputProbabilities;
		
		public RestrictedBoltzmannMachineFactory(RbmType rbmType, int visibleStatesCount, int hiddenStatesCount, 
			DistributionType startWeightGenerator, float[] inputProbabilities = null) {

			_rbmType = rbmType;
			_visibleStatesCount = visibleStatesCount;
			_hiddenStatesCount = hiddenStatesCount;
			_startWeightGenerator = startWeightGenerator;
			_inputProbabilities = inputProbabilities;
			if ((_inputProbabilities != null) && (_inputProbabilities.Length != visibleStatesCount)) {
				_inputProbabilities = null;
			}
		}
		
		public INeuralNet CreateNeuralNet() {
			var neuralNet = InstantiateRbm(_rbmType);
			if (neuralNet == null) {
				return null;
			}
			
			var random = new Random();

			var weights = neuralNet.Weights;
			switch (_startWeightGenerator) {
				case DistributionType.Null:
					for (var i = 0; i < weights.Length; i++) {
						weights[i] = 0.0f;
					}
					break;
				case DistributionType.Uniform:
					var factor = (float) (4.0*Math.Sqrt(6.0/(_visibleStatesCount + _hiddenStatesCount)));
					if (_inputProbabilities != null) {
						factor = 18.0f/(_visibleStatesCount + _hiddenStatesCount);
					}
					for (var i = 0; i < weights.Length; i++) {
						weights[i] = factor*(2.0f*(float)random.NextDouble() - 1.0f);
					}
					break;
				case DistributionType.Normal:
					var sigma = (float) Math.Sqrt(6.0/(_visibleStatesCount + _hiddenStatesCount));
					if (_inputProbabilities != null) {
						sigma = 6.0f/(_visibleStatesCount + _hiddenStatesCount);
					}
					float normal1, normal2;
					var length = weights.Length;
					if (length%2 != 0) {
						length--;
						GenerateNormal(random, out normal1, out normal2);
						weights[length] = sigma*normal1;
					}
					for (var i = 0; i < length; i += 2) {
						GenerateNormal(random, out normal1, out normal2);
						weights[i] = sigma*normal1;
						weights[i + 1] = sigma*normal2;
					}
					break;
			}

			var visibleStatesBias = neuralNet.VisibleStatesBias;
			if (_inputProbabilities != null) {
				var minBorderValue = float.MaxValue;
				var maxBorderValue = float.MinValue;
				for (var i = 0; i < visibleStatesBias.Length; i++) {
					var probability = _inputProbabilities[i];
					if ((Math.Abs(probability) > float.Epsilon) && (Math.Abs(1.0f - probability) > float.Epsilon)) {
						var value = (float) Math.Log(probability/(1.0 - probability));
						visibleStatesBias[i] = value;
						minBorderValue = Math.Min(minBorderValue, -Math.Abs(value));
						maxBorderValue = Math.Max(maxBorderValue, Math.Abs(value));
					}
				}
				for (var i = 0; i < visibleStatesBias.Length; i++) {
					var probability = _inputProbabilities[i];
					if (Math.Abs(probability) <= float.Epsilon) {
						visibleStatesBias[i] = minBorderValue - BiasStartValueBorder;
					}
					else if (Math.Abs(1.0f - probability) <= float.Epsilon) {
						visibleStatesBias[i] = maxBorderValue + BiasStartValueBorder;
					}
				}

			}
			else {
				for (var i = 0; i < visibleStatesBias.Length; i++) {
					visibleStatesBias[i] = 0.0f;
				}
			}
			
			var hiddenStatesBias = neuralNet.HiddenStatesBias;
			for (var i = 0; i < hiddenStatesBias.Length; i++) {
				hiddenStatesBias[i] = 0.0f;
			}

			return neuralNet;
		}

		private RestrictedBoltzmannMachine InstantiateRbm(RbmType rbmType) {
			RestrictedBoltzmannMachine rbm = null;
			switch (rbmType) {
				case RbmType.BinaryBinary:
					rbm = new BinaryBinaryRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType.BinaryNrelu:
					rbm = new BinaryNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType.GaussianBinary:
					rbm = new GaussianBinaryRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType.GaussianNrelu:
					rbm = new GaussianNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType.ReluNrelu:
					rbm = new ReluNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
			}
			return rbm;
		}

		private static void GenerateNormal(Random random, out float x1, out float x2) {
			double x, y;
			double s;
			do {
				x = 2.0*random.NextDouble() - 1.0;
				y = 2.0*random.NextDouble() - 1.0;
				s = x*x + y*y;
			} while ((s <= 0.0f) || (s > 1.0f));
			var factor = 1.0/s;
			factor = Math.Sqrt(2.0*factor*Math.Log(factor, Math.E));
			x1 = (float) (x*factor);
			x2 = (float) (y*factor);
		}
	}
}