using System;

namespace StandardTypes {
	public sealed class HammingDistance : IMetrics {
		public float Calculate(float[] realOtput, float[] reconstructedOutput) {
			var num = 0;
			for (var i = 0; i < realOtput.Length; i++) {
				if (Math.Abs(realOtput[i] - reconstructedOutput[i]) > float.Epsilon) {
					num++;
				}
			}
			return num;
		}

		public float[] CalculatePartialDerivaitve(float[] realOutput, float[] reconstructedOutput) {
			throw new NotImplementedException();
		}
	}
}