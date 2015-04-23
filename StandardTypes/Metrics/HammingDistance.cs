using System;

namespace StandardTypes {
	public sealed class HammingDistance : IMetrics {
		public float Calculate(float[] real, float[] reconstructed) {
			var num = 0;
			for (var i = 0; i < real.Length; i++) {
				if (Math.Abs(real[i] - reconstructed[i]) > float.Epsilon) {
					num++;
				}
			}
			return num;
		}

		public float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed) {
			throw new NotImplementedException();
		}
	}
}
