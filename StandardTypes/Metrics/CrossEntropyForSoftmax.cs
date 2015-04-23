using System;

namespace StandardTypes {
	public sealed class CrossEntropyForSoftmax : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var d = 0.0;
			for (var i = 0; i < real.Length; i++) {
				d += real[i]*Math.Log(reconstructed[i]);
			}
			return (float) -d;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			var result = new float[real.Length];
			for (var i = 0; i < real.Length; i++) {			
				result[i] = reconstructed[i] - real[i];
			}
			return result;
		}
	}
}
