#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include <immintrin.h>
#include "RestrictedBoltzmannMachine.h"
#include "malloc.h"
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <algorithm>

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		RestrictedBoltzmannMachineBase::RestrictedBoltzmannMachineBase(int visibleStatesCount, int hiddenStatesCount) {
			_randomDevice = new std::mt19937();
			_uniformDistribution = new std::uniform_real_distribution<float>(0.0f, 1.0f);
			_visibleStatesCount = visibleStatesCount;
			_hiddenStatesCount = hiddenStatesCount;
			_visibleStates = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_visibleStatesBias = (float*)_mm_malloc(_visibleStatesCount*sizeof(float), 32);
			_hiddenStates = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_hiddenStatesBias = (float*)_mm_malloc(_hiddenStatesCount*sizeof(float), 32);
			_weights = (float*)_mm_malloc(_visibleStatesCount*_hiddenStatesCount*sizeof(float), 32);
		}

		RestrictedBoltzmannMachineBase::~RestrictedBoltzmannMachineBase() {
			delete _randomDevice;
			delete _uniformDistribution;

			_mm_free(_visibleStates);
			_mm_free(_visibleStatesBias);
			_mm_free(_hiddenStates);
			_mm_free(_hiddenStatesBias);
			_mm_free(_weights);
		}
		
		void RestrictedBoltzmannMachineBase::VisibleLayerSampling(void) {
			for (int i = 0; i < _visibleStatesCount; i++) {
				_visibleStates[i] = (float) islessf((*_uniformDistribution)(*_randomDevice), _visibleStates[i]);
			}
		}

		void RestrictedBoltzmannMachineBase::HiddenLayerSampling(void) {
			for (int i = 0; i < _hiddenStatesCount; i++) {
				_hiddenStates[i] = (float) islessf((*_uniformDistribution)(*_randomDevice), _hiddenStates[i]);
			}
		}

		void RestrictedBoltzmannMachineBase::VisibleLayerCopyTo(float *target) {
            std::copy(_visibleStates, _visibleStates + _visibleStatesCount, target);
		}
		
		void RestrictedBoltzmannMachineBase::HiddenLayerCopyTo(float *target) {
			std::copy(_hiddenStates, _hiddenStates + _hiddenStatesCount, target);
		}
		
		void RestrictedBoltzmannMachineBase::Predict(const float *input, float *output) {
			HiddenLayerCalculateActivity(input);
			HiddenLayerSampling();
			VisibleLayerCalculateActivity();
			VisibleLayerSampling();
            VisibleLayerCopyTo(output);
		}

		int RestrictedBoltzmannMachineBase::GetVisibleStatesCount(void) {
			return _visibleStatesCount;
		}

		int RestrictedBoltzmannMachineBase::GetHiddenStatesCount(void) {
			return _hiddenStatesCount;
		}

		float* RestrictedBoltzmannMachineBase::GetWeights(void) {
			return _weights;
		}

		float* RestrictedBoltzmannMachineBase::GetVisibleStatesBias(void) {
			return _visibleStatesBias;
		}

		float* RestrictedBoltzmannMachineBase::GetHiddenStatesBias() {
			return _hiddenStatesBias;
		}

		float* RestrictedBoltzmannMachineBase::GetVisibleStates(void) {
			return _visibleStates;
		}

		float* RestrictedBoltzmannMachineBase::GetHiddenStates(void) {
			return _hiddenStates;
		}
	}
}