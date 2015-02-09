namespace StandardTypes {
	public interface IMetrics {
		float Calculate(float[] realOtput, float[] reconstructedOutput);
		float[] CalculatePartialDerivaitve(float[] realOutput, float[] reconstructedOutput);
	}
}