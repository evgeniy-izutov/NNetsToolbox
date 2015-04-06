using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public abstract class RestrictedBoltzmannMachine : INeuralNet {
	    private readonly Random uniformGenerator;
		protected float[] visibleStates;
		protected float[] hiddenStates;
		protected float[] weights;
		protected float[] visibleStatesBias;
		protected float[] hiddenStatesBias;

		protected RestrictedBoltzmannMachine() {
			uniformGenerator = new Random();
		}

		protected RestrictedBoltzmannMachine(int visibleStatesCount, int hiddenStatesCount) {
			uniformGenerator = new Random();
			visibleStates = new float[visibleStatesCount];
			visibleStatesBias = new float[visibleStatesCount];
			hiddenStates = new float[hiddenStatesCount];
			hiddenStatesBias = new float[hiddenStatesCount];
			weights = new float[visibleStatesCount*hiddenStatesCount];
		}

		public abstract void VisibleLayerCalculateActivity();

		public abstract void HiddenLayerCalculateActivity();

		public abstract void HiddenLayerCalculateActivity(float[] newVisibleState);

		public abstract void VisibleLayerCalculateActivity(float[] addedWeight, float[] addedVisibleBias);

		public abstract void HiddenLayerCalculateActivity(float[] addedWeight, float[] addedHiddenBias);

		public abstract void HiddenLayerCalculateActivity(float[] newVisibleState, float[] addedWeight, float[] addedHiddenBias);

		public virtual void VisibleLayerSampling() {
			for (var i = 0; i < visibleStates.Length; i++) {
				visibleStates[i] = uniformGenerator.NextDouble() < visibleStates[i] ? 1.0f : 0.0f;
			}
		}

		public virtual void HiddenLayerSampling() {
			for (var i = 0; i < hiddenStates.Length; i++) {
				hiddenStates[i] = uniformGenerator.NextDouble() < hiddenStates[i] ? 1.0f : 0.0f;
			}
		}

		public virtual void VisibleLayerSampling(float[] target) {
			for (var i = 0; i < visibleStates.Length; i++) {
				target[i] = uniformGenerator.NextDouble() < visibleStates[i] ? 1.0f : 0.0f;
			}
		}

		public virtual void HiddenLayerSampling(float[] target) {
			for (var i = 0; i < hiddenStates.Length; i++) {
				target[i] = uniformGenerator.NextDouble() < hiddenStates[i] ? 1.0f : 0.0f;
			}
		}

		public void CopyVisibleLayerTo(float[] target) {
			visibleStates.CopyTo(target, 0);
		}

		public void CopyHiddenLayerTo(float[] target) {
			hiddenStates.CopyTo(target, 0);
		}

		public void Predict(float[] input, float[] output) {
			Predict(input, output, true);
		}

		public float[] Predict(float[] input) {
			HiddenLayerCalculateActivity(input);
			HiddenLayerSampling();
			VisibleLayerCalculateActivity();
			VisibleLayerSampling();
			return (float[]) visibleStates.Clone();
		}

		public void Predict(float[] input, float[] output, bool isSamplingOutput) {
			HiddenLayerCalculateActivity(input);
			HiddenLayerSampling();
			VisibleLayerCalculateActivity();
			if (isSamplingOutput) {
				VisibleLayerSampling();
			}
			visibleStates.CopyTo(output, 0);
		}

		public void CalculateHiddenStates(float[] input, float[] output, bool isSamplingOutput) {
            HiddenLayerCalculateActivity(input);
			if (isSamplingOutput) {
		        HiddenLayerSampling();
	        }
			hiddenStates.CopyTo(output, 0);
	    }

		public byte[] SaveState() {
			byte[] bytes;
            IFormatter formatter = new BinaryFormatter();
            using (var stream = new MemoryStream()) {
                formatter.Serialize(stream, visibleStates);
			    formatter.Serialize(stream, hiddenStates);
			    formatter.Serialize(stream, weights);
			    formatter.Serialize(stream, visibleStatesBias);
			    formatter.Serialize(stream, hiddenStatesBias);
				
                bytes = stream.ToArray();
            }
            return bytes;
		}

		public void LoadState(byte[] state) {
			IFormatter formatter = new BinaryFormatter();
            using (var stream = new MemoryStream(state)) {			
				visibleStates = (float[]) formatter.Deserialize(stream);
			    hiddenStates = (float[]) formatter.Deserialize(stream);
			    weights = (float[]) formatter.Deserialize(stream);
			    visibleStatesBias = (float[]) formatter.Deserialize(stream);
			    hiddenStatesBias = (float[]) formatter.Deserialize(stream);
            }
		}

		public float[] Weights {
			get { return weights; }
		}

		public float[] VisibleStatesBias {
			get { return visibleStatesBias; }
		}

		public float[] HiddenStatesBias {
			get { return hiddenStatesBias; }
		}

		public float[] VisibleStates {
			get { return visibleStates; }
		}

		public float[] HiddenStates {
			get { return hiddenStates; }
		}
	}
}