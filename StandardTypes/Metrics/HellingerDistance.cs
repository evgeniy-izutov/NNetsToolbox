using System;

namespace StandardTypes {
	public sealed class HellingerDistance : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var value = 1f;
			for (var i = 0; i < real.Length; i++) {
				value -= (float) Math.Sqrt(real[i]*reconstructed[i]);
			}
			return 2f*value;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			var result = new float[real.Length];
			for (var i = 0; i < real.Length; i++) {
				result[i] = 1f - (float) Math.Sqrt(real[i]/reconstructed[i]);
			}
			return result;
		}
	}
}
