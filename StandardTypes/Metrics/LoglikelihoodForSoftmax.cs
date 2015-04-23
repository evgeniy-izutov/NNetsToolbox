using System;

namespace StandardTypes {
	public sealed class LoglikelihoodForSoftmax : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var d = 0d;
			var length = real.Length;
			for (var i = 0; i < length; i++) {
				d += real[i]*Math.Log(reconstructed[i]) + (1f - real[i])*Math.Log(1f - reconstructed[i]);
			}
			return (float) -d;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			var sum = 0f;
			for (var i = 0; i < real.Length; i++) {
				sum += reconstructed[i]*(1f - real[i])/(1f - reconstructed[i]);
			}

			var result = new float[real.Length];
			for (var i = 0; i < real.Length; i++) {
				result[i] = reconstructed[i]*(2f - real[i] + 
					sum - reconstructed[i]*(1f - real[i])/(1f - reconstructed[i])) - real[i];
			}
			return result;
		}
	}
}
