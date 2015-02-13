#pragma once

#include "ExportDll.h"
#include "TrainSingle.h"
#include "TrainPair.h"
#include <random>

namespace StandardTypesNative {
	template<typename T>
	class STANDARDTYPES_EXPORT RandomAccessIterator {
	private:
		int _size;
		int *_positions;
		T *_sourceList;
		std::random_device *_randomDevice;
		std::uniform_int_distribution<int> *_uniformDistribution;
		int _lastRandomAccessIndex;
	public:
		RandomAccessIterator(T *list, int size);
		~RandomAccessIterator(void);
		void RefreshRandomAccess(void);
		T& Next(void);
		int Size(void) const;
		T* Collection(void) const;
	private:
		static int* CreateStartPositions(int size);
	};

	template class STANDARDTYPES_EXPORT RandomAccessIterator<TrainSingle>;
	template class STANDARDTYPES_EXPORT RandomAccessIterator<TrainPair>;
	template class STANDARDTYPES_EXPORT RandomAccessIterator<TrainSingle*>;
	template class STANDARDTYPES_EXPORT RandomAccessIterator<TrainPair*>;
}