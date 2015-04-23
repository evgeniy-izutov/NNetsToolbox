using System;
using System.Collections.Generic;

namespace StandardTypes.SetWeights {
	public sealed class ExponentialWeights<T> : ISetWeightsAdaptation<T> where T:TrainData {
		private readonly double _weightFactor;

		public ExponentialWeights(float weightFactor) {
			_weightFactor = weightFactor;
		}

		public void ChangeWeights(List<T> set, float[] errors) {
			var sumWeights = 0d;
			for (var i = 0; i < errors.Length; i++) {
				var example = set[i];
				
				var newWeight = example.Weight*Math.Exp(_weightFactor*errors[i]);
				sumWeights += newWeight;
				example.Weight = (float) newWeight;
			}

			for (var i = 0; i < set.Count; i++) {
				set[i].Weight /= (float) sumWeights;
			}
		}
	}
}
