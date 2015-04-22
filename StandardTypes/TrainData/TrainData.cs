using System;

namespace StandardTypes {
    [Serializable]
    public abstract class TrainData {
		public int Id { get; set; }
		public float Weight { get; set; }

	    protected TrainData() {
		    Weight = 1f;
	    }

	    protected TrainData(TrainData source) {
		    Id = source.Id;
			Weight = source.Weight;
	    }
    }
}
