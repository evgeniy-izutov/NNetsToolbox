using System;

namespace StandardTypes {
	public sealed class CrossEntropyForSoftmax : IMetrics {
		public float Calculate(float[] realOtput, float[] reconstructedOutput) {
			var d = 0.0;
			for (var i = 0; i < realOtput.Length; i++) {
				d += realOtput[i]*Math.Log(reconstructedOutput[i]);
			}
			return (float) -d;
		}

		public float[] CalculatePartialDerivaitve(float[] realOutput, float[] reconstructedOutput) {
			var result = new float[realOutput.Length];
			for (var i = 0; i < realOutput.Length; i++) {			
				result[i] = reconstructedOutput[i] - realOutput[i];
			}
			return result;
		}
	}
}