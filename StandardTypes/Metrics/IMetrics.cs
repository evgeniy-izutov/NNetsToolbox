namespace StandardTypes {
	public interface IMetrics {
		float Calculate(float[] real, float[] reconstructed);
		float[] CalculatePartialDerivaitve(float[] real, float[] reconstructed);
	}
}