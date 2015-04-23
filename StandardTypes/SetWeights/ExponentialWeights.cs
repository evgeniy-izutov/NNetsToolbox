using System;
using System.Collections.Generic;
using StandardTypes.FactorStrategy;

namespace StandardTypes.SetWeights {
	public sealed class ExponentialWeights<T> : ISetWeightsAdaptation<T> where T:TrainData {
		private readonly IFactorStrategy _factorStrategy;

		public ExponentialWeights(IFactorStrategy factorStrategy) {
			_factorStrategy = factorStrategy;
		}

		public void ChangeWeights(IList<T> set, float[] errors) {
			var sumWeights = 0d;
			for (var i = 0; i < errors.Length; i++) {
				var example = set[i];
				
				var newWeight = example.Weight*Math.Exp(_factorStrategy.GetFactor()*errors[i]);
				sumWeights += newWeight;
				example.Weight = (float) newWeight;
			}

			for (var i = 0; i < set.Count; i++) {
				set[i].Weight /= (float) sumWeights;
			}
		}
	}
}
