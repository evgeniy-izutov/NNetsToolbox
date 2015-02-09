#define STANDARDTYPESAPI
#include "RandomAccessIterator.h"
#include <random>

namespace StandardTypesNative {
	template<typename T>
	RandomAccessIterator<T>::RandomAccessIterator(T *list, int size) {
		_randomDevice = new std::random_device();
		_uniformDistribution = new std::uniform_int_distribution<int>(0, size - 1);
		_size = size;
		_sourceList = list;
		_positions = CreateStartPositions(size);
		RefreshRandomAccess();
	}

	template<typename T>
	RandomAccessIterator<T>::~RandomAccessIterator(void) {
		_size = 0;
		delete _randomDevice;
		delete _uniformDistribution;
		delete [] _positions;
	}

	template<typename T>
	void RandomAccessIterator<T>::RefreshRandomAccess(void) {
		for (int i = _size - 1; i > 0; i--) {
			int newIndex = (*_uniformDistribution)(*_randomDevice);
			if (newIndex != i) {
				int tmp = _positions[newIndex];
				_positions[newIndex] = _positions[i];
				_positions[i] = tmp;
			}
		}
		_lastRandomAccessIndex = 0;
	}

	template<typename T>
	T& RandomAccessIterator<T>::Next(void) {
		if (_lastRandomAccessIndex >= _size) {
			RefreshRandomAccess();
		}
		return _sourceList[_positions[_lastRandomAccessIndex++]];
	}

	template<typename T>
	int RandomAccessIterator<T>::Size(void) {
		return _size;
	}

	template<typename T>
	int* RandomAccessIterator<T>::CreateStartPositions(int size) {
		int *positions = new int[size];
		for (int i = 0; i < size; i++) {
			positions[i] = i;
		}
		return positions;
	}

	template<typename T>
	T* RandomAccessIterator<T>::Collection(void) {
		return _sourceList;
	}
}