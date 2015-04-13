using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNet.ClassificationRbm {
	public sealed class ClassificationRbm : INeuralNet {
		private readonly Random _uniformGenerator;
		private float[] _visibleStates;
		private float[] _hiddenStates;
		private float[] _visibleStatesWeights;
		private float[] _visibleStatesBias;
		private float[] _hiddenStatesBias;
		private float[] _labels;
		private float[] _labelsBias;
		private float[] _labelsWeights;
		private double[] _temporarySums;
		private double[] _temporaryPredictions;

		public ClassificationRbm() {
			_uniformGenerator = new Random();
		}

		public ClassificationRbm(int visibleStatesCount, int hiddenStatesCount, int labelsCount) {
			_uniformGenerator = new Random();
			_visibleStates = new float[visibleStatesCount];
			_visibleStatesBias = new float[visibleStatesCount];
			_hiddenStates = new float[hiddenStatesCount];
			_hiddenStatesBias = new float[hiddenStatesCount];
			_visibleStatesWeights = new float[visibleStatesCount*hiddenStatesCount];
			_labels = new float[labelsCount];
			_labelsBias = new float[labelsCount];
			_labelsWeights = new float[labelsCount*hiddenStatesCount];
			_temporarySums = new double[hiddenStatesCount];
			_temporaryPredictions = new double[labelsCount];
		}

		public void InputLayerCalculateActivity() {
			VisibleLayerCalculateActivity();
			LabelLayerCalculateActivity();
		}
		
		public void VisibleLayerCalculateActivity() {
			for (var i = 0; i < _visibleStates.Length; i++) {
				_visibleStates[i] = _visibleStatesBias[i];
			}

			for (var j = 0; j < _hiddenStates.Length; j++) {
				var hiddenState = _hiddenStates[j];
				var weightsStartPos = j*_visibleStates.Length;
				for (var i = 0; i < _visibleStates.Length; i++) {
					_visibleStates[i] += hiddenState*_visibleStatesWeights[weightsStartPos + i];
				}
			}

			for (var i = 0; i < _visibleStates.Length; i++) {
				_visibleStates[i] = 1f/(1f + (float) Math.Exp(-_visibleStates[i]));
			}
		}

		public void LabelLayerCalculateActivity() {		
			for (var i = 0; i < _labels.Length; i++) {
				_labels[i] = _labelsBias[i];
			}

			for (var j = 0; j < _hiddenStates.Length; j++) {
				var hiddenState = _hiddenStates[j];
				var weightsStartPos = j*_labels.Length;
				for (var i = 0; i < _labels.Length; i++) {
					_labels[i] += hiddenState*_labelsWeights[weightsStartPos + i];
				}
			}

			var sum = 0f;
			for (var i = 0; i < _labels.Length; i++) {
				_labels[i] = 1f/(1f + (float) Math.Exp(-_labels[i]));
				sum += _labels[i];
			}

			for (var i = 0; i < _labels.Length; i++) {
				_labels[i] /= sum;
			}
		}

		public void HiddenLayerCalculateActivity() {
			HiddenLayerCalculateActivity(_visibleStates, _labels);
		}

		public void HiddenLayerSampling() {
			for (var i = 0; i < _hiddenStates.Length; i++) {
				_hiddenStates[i] = _uniformGenerator.NextDouble() < _hiddenStates[i] ? 1.0f : 0.0f;
			}
		}

		public void HiddenLayerCalculateActivity(float[] newVisibleState, float[] newLabels) {
			for (var j = 0; j < _hiddenStates.Length; j++) {
				var sum = _hiddenStatesBias[j];

				var weightsStartPos = j*newVisibleState.Length;
				for (var k = 0; k < newVisibleState.Length; k++) {
					sum += newVisibleState[k]*_visibleStatesWeights[weightsStartPos + k];
				}

				weightsStartPos = j*newLabels.Length;
				for (var k = 0; k < newLabels.Length; k++) {
					sum += newLabels[k]*_labelsWeights[weightsStartPos + k];
				}

				_hiddenStates[j] = 1f/(1f + (float) Math.Exp(-sum));
			}
		}

		public float[] Predict(float[] input) {
			var prediction = new float[_labels.Length];
			Predict(input, prediction);
			return prediction;
		}

		public void Predict(float[] input, float[] output) {		
			double sum;
			for (var j = 0; j < _hiddenStates.Length; j++) {
				sum = _hiddenStatesBias[j];
				
				var weightsStartPos = j*_visibleStates.Length;
				for (var k = 0; k < _visibleStates.Length; k++) {
					sum += input[k]*_visibleStatesWeights[weightsStartPos + k];
				}

				_temporarySums[j] = sum;
			}

			for (var k = 0; k < _labels.Length; k++) {
				_temporaryPredictions[k] = (float) Math.Exp(_labelsBias[k]);
			}

			for (var j = 0; j < _hiddenStates.Length; j++) {
				var temporarySum = _temporarySums[j];
				for (var k = 0; k < _labels.Length; k++) {
					_temporaryPredictions[k] *= 1d + Math.Exp(temporarySum + _labelsWeights[j*_labels.Length + k]);
				}
			}

			sum = 0d;
			for (var k = 0; k < _labels.Length; k++) {
				sum += _temporaryPredictions[k];
			}

			for (var k = 0; k < _labels.Length; k++) {
				output[k] = (float) (_temporaryPredictions[k]/sum);
			}
		}

		public byte[] SaveState() {
			byte[] bytes;
            IFormatter formatter = new BinaryFormatter();
            using (var stream = new MemoryStream()) {
			    formatter.Serialize(stream, _visibleStatesWeights);
			    formatter.Serialize(stream, _visibleStatesBias);
			    formatter.Serialize(stream, _hiddenStatesBias);
				formatter.Serialize(stream, _labelsBias);
				formatter.Serialize(stream, _labelsWeights);
				
                bytes = stream.ToArray();
            }
            return bytes;
		}

		public void LoadState(byte[] state) {
			IFormatter formatter = new BinaryFormatter();
            using (var stream = new MemoryStream(state)) {			
			    _visibleStatesWeights = (float[]) formatter.Deserialize(stream);
			    _visibleStatesBias = (float[]) formatter.Deserialize(stream);
			    _hiddenStatesBias = (float[]) formatter.Deserialize(stream);
				_labelsBias = (float[]) formatter.Deserialize(stream);
	            _labelsWeights = (float[]) formatter.Deserialize(stream);
            }

			_visibleStates = new float[_visibleStatesBias.Length];
			_hiddenStates = new float[_hiddenStatesBias.Length];
			_labels = new float[_labelsBias.Length];
			_temporarySums = new double[_hiddenStatesBias.Length];
			_temporaryPredictions = new double[_labelsBias.Length];
		}

		public float[] VisibleStatesWeights {
			get { return _visibleStatesWeights; }
		}

		public float[] VisibleStatesBias {
			get { return _visibleStatesBias; }
		}

		public float[] HiddenStatesBias {
			get { return _hiddenStatesBias; }
		}

		public float[] VisibleStates {
			get { return _visibleStates; }
		}

		public float[] HiddenStates {
			get { return _hiddenStates; }
		}

		public float[] Labels {
			get { return _labels; }
		}

		public float[] LabelsBias {
			get { return _labelsBias; }
		}

		public float[] LabelsWeights {
			get { return _labelsWeights; }
		}
	}
}
