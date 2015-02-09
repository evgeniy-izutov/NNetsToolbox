using System;

namespace StandardTypes {
	public enum FeatureType {
		Value,
		Categorial,
		Id,
		Weight
	}
	
	[Serializable]
	public sealed class FeatureDescription {
		public int SourceId { get; set; }
		public FeatureType Type { get; set; }
		public int CategoriesCount { get; set; }
	}
}