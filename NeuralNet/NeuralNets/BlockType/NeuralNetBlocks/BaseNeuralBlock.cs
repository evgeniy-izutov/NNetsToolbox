using System;
using NeuralNet.ActivationFunctions;

namespace NeuralNet.NeuralNetBlocks {
    [Serializable]
    public abstract class BaseNeuralBlock {
        protected BaseNeuralBlock[] Parents;
        protected float[][] Weights;
        protected float[] Bias;
        protected IActivationFunction ActivationFunction;
        protected float[] State;
        protected float[] Net;

        protected BaseNeuralBlock(int size, BaseNeuralBlock[] parents, IActivationFunction activationFunction) {
            ActivationFunction = activationFunction;
            Bias = new float[size];
            State = new float[size];
            Net = new float[size];
            Parents = new BaseNeuralBlock[parents.Length];
            Weights = new float[parents.Length][];
            for (var i = 0; i < parents.Length; i++) {
                var parent = parents[i];
                Parents[i] = parent;
                Weights[i] = new float[size*parent.Size];
            }
        }

        protected BaseNeuralBlock(int size, int parentSize, IActivationFunction activationFunction) {
            ActivationFunction = activationFunction;
            Bias = new float[size];
            State = new float[size];
            Net = new float[size];
            Parents = new BaseNeuralBlock[] {null};
            Weights = new[] {new float[size*parentSize]};
        }

        public float[] GetState() {
            return State;
        }

        public float[] GetNet() {
            return Net;
        }

        public BaseNeuralBlock[] GetParents() {
            return Parents;
        }

        public float[][] GetWeights() {
            return Weights;
        }

        public float[] GetBias() {
            return Bias;
        }

        public IActivationFunction GetActivationFunction() {
            return ActivationFunction;
        }

		public void SetWeightsFor(int num, float[] newWeights) {
            Weights[num] = newWeights;
        }

		public void SetBias(float[] newBias) {
			Bias = newBias;
		}

        public int Size {
            get { return State.Length; }
        }

    	public IActivationFunction Function {
    		get { return ActivationFunction; }
    	}

        public abstract void Calculate();

        public abstract void Calculate(float[] input);
    }
}
