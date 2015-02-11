using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNet.RestrictedBoltzmannMachine {
	public abstract class RestrictedBoltzmannMachine : INeuralNet {
		protected readonly Random uniformGenerator;
		protected float[] visibleStates;
		protected float[] hiddenStates;
		protected float[] weights;
		protected float[] visibleStatesBias;
		protected float[] hiddenStatesBias;

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
			for (var i = 0; i < visibleStates.Length; i++) {
				target[i] = visibleStates[i];
			}
		}

		public void CopyHiddenLayerTo(float[] target) {
			for (var i = 0; i < hiddenStates.Length; i++) {
				target[i] = hiddenStates[i];
			}
		}
		
		public void Predict(float[] input, float[] output) {
			HiddenLayerCalculateActivity(input);
			HiddenLayerSampling();
			VisibleLayerCalculateActivity();
			VisibleLayerSampling();
			CopyVector(visibleStates, output);
		}

		public void Predict(float[] input, float[] output, bool isSamplingOutput) {
			HiddenLayerCalculateActivity(input);
			HiddenLayerSampling();
			VisibleLayerCalculateActivity();
			if (isSamplingOutput) {
				VisibleLayerSampling();
			}
			CopyVector(visibleStates, output);
		}

        public void CalculateHiddenStates(float[] input, float[] output, bool isSamplingOutput){
            HiddenLayerCalculateActivity(input);
			if (isSamplingOutput) {
		        HiddenLayerSampling();
	        }
			CopyVector(hiddenStates, output);
	    }

		public void Save(string outputPath) {
			var outputStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
			var serializer = new BinaryFormatter();
			serializer.Serialize(outputStream, visibleStates);
			serializer.Serialize(outputStream, hiddenStates);
			serializer.Serialize(outputStream, weights);
			serializer.Serialize(outputStream, visibleStatesBias);
			serializer.Serialize(outputStream, hiddenStatesBias);
			outputStream.Close();
		}

		public void Load(string inputPath) {
			var inputStream = new FileStream(inputPath, FileMode.Open, FileAccess.Read, FileShare.Read);
			var deserializer = new BinaryFormatter();
			visibleStates = (float[]) deserializer.Deserialize(inputStream);
			hiddenStates = (float[]) deserializer.Deserialize(inputStream);
			weights = (float[]) deserializer.Deserialize(inputStream);
			visibleStatesBias = (float[]) deserializer.Deserialize(inputStream);
			hiddenStatesBias = (float[]) deserializer.Deserialize(inputStream);
			inputStream.Close();
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

		private static void CopyVector(float[] source, float[] target) {
            for (var i = 0; i < source.Length; i++) {
                target[i] = source[i];
            }
        }
	}
}