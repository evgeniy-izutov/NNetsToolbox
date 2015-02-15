#define NEURALNETNATIVEAPI
#include <mathimf.h>
#include <immintrin.h>
#include <tbb\tbb.h>
#include <tbb\task_scheduler_init.h>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <algorithm>
#include "BinaryBinaryRbm.h"

using namespace tbb;

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		BinaryBinaryRbm::BinaryBinaryRbm(int visibleStatesCount, int hiddenStatesCount) : RestrictedBoltzmannMachineBase(visibleStatesCount, hiddenStatesCount) {
		}
		
		void BinaryBinaryRbm::VisibleLayerCalculateActivity(void) {
            std::copy(_visibleStatesBias, _visibleStatesBias + _visibleStatesCount, _visibleStates);

			for (int j = 0; j < _hiddenStatesCount; j++) {
				float hiddenState = _hiddenStates[j];
				for (int i = 0; i < _visibleStatesCount; i++) {
					_visibleStates[i] += hiddenState*_weights[j*_visibleStatesCount + i];
				}
			}

			for (int i = 0; i < _visibleStatesCount; i++) {
				_visibleStates[i] = 1.0f/(1.0f + expf(-_visibleStates[i]));
			}
		}

		void BinaryBinaryRbm::HiddenLayerCalculateActivity(void) {
			parallel_for( blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float sum = _hiddenStatesBias[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						sum += _visibleStates[i]*_weights[j*_visibleStatesCount + i];
					}
					_hiddenStates[j] = 1.0f/(1.0f + expf(-sum));
				}
			});
		}

		void BinaryBinaryRbm::HiddenLayerCalculateActivity(const float *newVisibleState) {
			parallel_for( blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float sum = _hiddenStatesBias[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						sum += newVisibleState[i]*_weights[j*_visibleStatesCount + i];
					}
					_hiddenStates[j] = 1.0f/(1.0f + expf(-sum));
				}
			});
		}

		void BinaryBinaryRbm::VisibleLayerCalculateActivity(const float *addedWeight, const float *addedVisibleBias) {
			for (int i = 0; i < _visibleStatesCount; i++) {
				_visibleStates[i] = _visibleStatesBias[i] + addedVisibleBias[i];
			}

			for (int j = 0; j < _hiddenStatesCount; j++) {
				float hiddenState = _hiddenStates[j];
				for (int i = 0; i < _visibleStatesCount; i++) {
					_visibleStates[i] += hiddenState*(_weights[j*_visibleStatesCount + i] +
                                         addedWeight[j*_visibleStatesCount + i]);
				}
			}

			for (int i = 0; i < _visibleStatesCount; i++) {
				_visibleStates[i] = 1.0f/(1.0f + expf(-_visibleStates[i]));
			}
		}

		void BinaryBinaryRbm::HiddenLayerCalculateActivity(const float *addedWeight, const float *addedHiddenBias) {
			parallel_for( blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float sum = _hiddenStatesBias[j] + addedHiddenBias[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						sum += _visibleStates[i]*(_weights[j*_visibleStatesCount + i] +
                               addedWeight[j*_visibleStatesCount + i]);
					}
					_hiddenStates[j] = 1.0f/(1.0f + expf(-sum));
				}
			});
		}

		void BinaryBinaryRbm::HiddenLayerCalculateActivity(const float *newVisibleState, const float *addedWeight, const float *addedHiddenBias) {
			parallel_for( blocked_range<size_t>(0, _hiddenStatesCount),
			[=](const blocked_range<size_t>& r)
			{
				for (int j = r.begin(); j < r.end(); j++) {
					float sum = _hiddenStatesBias[j] + addedHiddenBias[j];
					for (int i = 0; i < _visibleStatesCount; i++) {
						sum += newVisibleState[i]*(_weights[j*_visibleStatesCount + i] +
                               addedWeight[j*_visibleStatesCount + i]);
					}
					_hiddenStates[j] = 1.0f/(1.0f + expf(-sum));
				}
			});
		}
	}
}