﻿namespace StandardTypes {
	public sealed class SquaredEuclidianDistance : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var value = 0.0f;
			for (var i = 0; i < real.Length; i++) {
				var dif = real[i] - reconstructed[i];
				value += dif*dif;
			}
			return 0.5f*value;
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
