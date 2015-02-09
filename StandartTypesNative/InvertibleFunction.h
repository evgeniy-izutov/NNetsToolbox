#pragma once

namespace StandardTypesNative {
	class InvertibleFunction {
	public:
		virtual float Calculate(float x) = 0;
		virtual float CalculateInvers(float y) = 0;
	};
}