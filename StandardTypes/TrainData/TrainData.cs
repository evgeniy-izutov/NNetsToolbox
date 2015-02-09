using System;

namespace StandardTypes {
    [Serializable]
    public abstract class TrainData {
		public int Id { get; set; }
		public float Weight { get; set; }
    }
}