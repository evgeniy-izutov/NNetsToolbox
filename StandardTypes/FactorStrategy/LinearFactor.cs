namespace StandardTypes.FactorStrategy {
    public sealed class LinearFactor : IFactorStrategy {
        private readonly float _a;
        private readonly float _b;
	    private int _localIterNumber;

        public LinearFactor(float startFactor, float endFactor, int stepsCount) {
            if (stepsCount <= 1) {
                _a = startFactor;
                _b = 0;
            }
            else {
                _a = (endFactor - startFactor)/(stepsCount - 1);
                _b = startFactor - _a;
            }

	        _localIterNumber = 1;
        }

        public LinearFactor(float a, float b) {
            _a = a;
            _b = b;
        }

        public float A {
            get { return _a; }
        }

        public float B {
            get { return _b; }
        }
        
        public float GetFactor(int iterNumber) {
            return _a*iterNumber + _b;
        }

	    public float GetFactor() {
		    var value = _a*_localIterNumber + _b;
		    _localIterNumber++;
		    return value;
	    }
    }
}
