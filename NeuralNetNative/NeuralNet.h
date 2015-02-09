#pragma once

namespace NeuralNetNative {
	class NeuralNet {
		virtual void Predict(const float *input, float *output) = 0;
	};
}