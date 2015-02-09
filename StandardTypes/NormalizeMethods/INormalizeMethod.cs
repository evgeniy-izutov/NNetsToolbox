using System.Collections.Generic;

namespace StandardTypes {
    public interface INormalizeMethod {
    	void CollectStatistics(IList<TrainSingle> data);
		void CollectStatistics(IList<TrainPair> data);
	    void FillRandomlyGaps(IList<TrainSingle> data);
		void FillRandomlyGaps(IList<TrainPair> data);
		void NormalizeSet(IList<TrainSingle> data, bool isSkipGaps);
    	void NormalizeSet(IList<TrainPair> data, bool isNormalizeOutput, bool isSkipGaps);
        float[] NormalizeInputVector(float[] inputVector, HashSet<int> missedInputIndexes);
	    float[] DenormalizeInputVector(float[] normalizedInputVector, HashSet<int> missedInputIndexes);
        void DenormalizeOutputVector(float[] outputVector);
        void Save(string path);
        void Load(string path);
        int InputVectorSize { get; }
	    int OutputVectorSize { get; }
    }
}