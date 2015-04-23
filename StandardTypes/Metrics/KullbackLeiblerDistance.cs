using System;

namespace StandardTypes {
	public sealed class KullbackLeiblerDistance : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var error = 0d;
			for (var i = 0; i < real.Length; i++) {
				if (Math.Abs(real[i]) > float.Epsilon)
					if (Math.Abs(reconstructed[i]) > float.Epsilon)
						error += real[i]*Math.Log(real[i]/reconstructed[i]);
			}
			return (float) error;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			var result = new float[real.Length];
			for (var i = 0; i < real.Length; i++) {
				result[i] = real[i]/reconstructed[i];
			}
			return result;
		}
	}
}
