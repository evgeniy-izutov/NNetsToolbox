using System;

namespace StandardTypes {
	public sealed class LoglikelihoodForSoftmax : IMetrics {
		public float Calculate(float[] realOtput, float[] reconstructedOutput) {
			var d = 0.0;
			var length = realOtput.Length;
			for (var i = 0; i < length; i++) {
				d += realOtput[i]*Math.Log(reconstructedOutput[i]) + (1.0f - realOtput[i])*Math.Log(1.0f - reconstructedOutput[i]);
			}
			return (float) -d;
		}

		public float[] CalculatePartialDerivaitve(float[] realOutput, float[] reconstructedOutput) {
			var sum = 0.0f;
			for (var i = 0; i < realOutput.Length; i++) {
				sum += reconstructedOutput[i]*(1.0f - realOutput[i])/(1.0f - reconstructedOutput[i]);
			}

			var result = new float[realOutput.Length];
			for (var i = 0; i < realOutput.Length; i++) {
				result[i] = reconstructedOutput[i]*(2.0f - realOutput[i] + 
					sum - reconstructedOutput[i]*(1.0f - realOutput[i])/(1.0f - reconstructedOutput[i])) - realOutput[i];
			}
			return result;
		}
	}
}