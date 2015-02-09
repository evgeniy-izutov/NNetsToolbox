namespace StandardTypes {
	public sealed class SquaredEuclidianDistance : IMetrics {
		public float Calculate(float[] realOtput, float[] reconstructedOutput) {
			var value = 0.0f;
			for (var i = 0; i < realOtput.Length; i++) {
				var dif = realOtput[i] - reconstructedOutput[i];
				value += dif*dif;
			}
			return 0.5f*value;
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