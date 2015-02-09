using StandardTypes;

namespace NeuralNet {
    public interface IActivationFunction : IInvertibleFunction {
        float CalculateFirstDerivative(float x);
		float CalculateFirstDerivative(float[] state, int index);
	    void CalculateFirstDerivative(float[] target, float[] factors, float[] state);
	    void CalculateFirstDerivative(float[] target, float[] state);
	    float GetMaxDerivativeZone(float maxValuePercent);
    }
}