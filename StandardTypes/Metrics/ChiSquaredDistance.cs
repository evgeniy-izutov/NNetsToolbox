namespace StandardTypes {
	public sealed class ChiSquaredDistance : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var value = 0.0f;
			for (var i = 0; i < real.Length; i++) {
				var dif = real[i] - reconstructed[i];
				value += dif*dif/real[i];
			}
			return value;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			var result = new float[real.Length];
			for (var i = 0; i < real.Length; i++) {
				result[i] = 2f*(reconstructed[i]/real[i] - 1f);
			}
			return result;
		}
	}
}
