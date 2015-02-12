#pragma once

namespace NeuralNetNative {
	class LearnFactorStrategy {
	public:
		virtual float GetFactor(int iterNumber) const = 0;
	};
}