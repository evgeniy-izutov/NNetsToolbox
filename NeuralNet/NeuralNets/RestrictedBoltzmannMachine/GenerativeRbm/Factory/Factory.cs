using System;
using MathNet.Numerics.Distributions;
using StandardTypes;

namespace NeuralNet.GenerativeRbm {
	public sealed class Factory : INeuralNetFactory {
		private const float BiasStartValueBorder = 20.0f; //empirical only
		private readonly int _visibleStatesCount;
		private readonly int _hiddenStatesCount;
		private readonly DistributionType _startWeightGenerator;
		private readonly RbmType _rbmType;
		private readonly float[] _inputProbabilities;
		private static readonly Normal NormalGenerator;
		private static readonly Random UniformGenerator;

		static Factory() {
			UniformGenerator = new Random();
			NormalGenerator = new Normal {
				RandomSource = new Random()
			};
		}
		
		public Factory(RbmType rbmType, int visibleStatesCount, int hiddenStatesCount, 
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

			switch (_startWeightGenerator) {
				case DistributionType.Null:
					FillZero(neuralNet.Weights);
					break;
				case DistributionType.Uniform:
					FillUniform(neuralNet.Weights, _visibleStatesCount, _hiddenStatesCount, _inputProbabilities);
					break;
				case DistributionType.Normal:
					FillNormal(neuralNet.Weights, _visibleStatesCount, _hiddenStatesCount, _inputProbabilities);
					break;
			}

			FillBias(neuralNet.VisibleStatesBias, _inputProbabilities);
			
			FillZero(neuralNet.HiddenStatesBias);

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

		private static void FillZero(float[] array) {
			for (var i = 0; i < array.Length; i++) {
				array[i] = 0f;
			}
		}

		private static void FillUniform(float[] array, int rowsCount, int columnsCount, float[] probabilities) {
			var factor = (float) (4d*Math.Sqrt(6d/(rowsCount + columnsCount)));
			if (probabilities != null) {
				factor = 18f/(rowsCount + columnsCount);
			}
			factor = Math.Min(0.03f, factor);

			for (var i = 0; i < array.Length; i++) {
				array[i] = factor*(2f * (float)UniformGenerator.NextDouble() - 1f);
			}
		}

		private static void FillNormal(float[] array, int rowsCount, int columnsCount, float[] probabilities) {
			var sigma = (float) Math.Sqrt(6d/(rowsCount + columnsCount));
			if (probabilities != null) {
				sigma = 6f/(rowsCount + columnsCount);
			}
			sigma = Math.Min(0.01f, sigma);

			for (var i = 0; i < array.Length; i++) {
				array[i] = sigma * (float) NormalGenerator.Sample();
			}
		}

		private static void FillBias(float[] bias, float[] probabilities) {
			if (probabilities != null) {
				var minBorderValue = float.MaxValue;
				var maxBorderValue = float.MinValue;
				for (var i = 0; i < bias.Length; i++) {
					var probability = probabilities[i];
					if ((Math.Abs(probability) > float.Epsilon) && (Math.Abs(1.0f - probability) > float.Epsilon)) {
						var value = (float) Math.Log(probability/(1.0 - probability));
						bias[i] = value;
						minBorderValue = Math.Min(minBorderValue, -Math.Abs(value));
						maxBorderValue = Math.Max(maxBorderValue, Math.Abs(value));
					}
				}
				for (var i = 0; i < bias.Length; i++) {
					var probability = probabilities[i];
					if (Math.Abs(probability) <= float.Epsilon) {
						bias[i] = minBorderValue - BiasStartValueBorder;
					}
					else if (Math.Abs(1.0f - probability) <= float.Epsilon) {
						bias[i] = maxBorderValue + BiasStartValueBorder;
					}
				}

			}
			else {
				FillZero(bias);
			}
		}
	}
}
