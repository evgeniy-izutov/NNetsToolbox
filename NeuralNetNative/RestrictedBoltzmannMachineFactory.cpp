#define NEURALNETNATIVEAPI
#include "RestrictedBoltzmannMachineFactory.h"
#include "BinaryBinaryRbm.h"
#include "GaussianBinaryRbm.h"
#include <random>

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		RestrictedBoltzmannMachineFactory::RestrictedBoltzmannMachineFactory(RbmType rbmType, int visibleStatesCount, int hiddenStatesCount, StartWeightGenerator startWeightGenerator) {
			_visibleStatesCount = visibleStatesCount;
			_hiddenStatesCount = hiddenStatesCount;
			_startWeightGenerator = startWeightGenerator;
			_rbmType = rbmType;
		}

		RestrictedBoltzmannMachineFactory::~RestrictedBoltzmannMachineFactory(void) {
			_visibleStatesCount = 0;
			_hiddenStatesCount = 0;
		}
		
		NeuralNet* RestrictedBoltzmannMachineFactory::CreateNeuralNet(void) {
			RestrictedBoltzmannMachineBase* neuralNet = InstantiateRbm(_rbmType);
			if (neuralNet == 0) {
				return 0;
			}
			
			std::random_device rd;
			std::mt19937 gen(rd());

			int visibleStatesCount = neuralNet->GetVisibleStatesCount();
			int hiddenStatesCount = neuralNet->GetHiddenStatesCount();
			int weightsCount = visibleStatesCount*hiddenStatesCount;
			float* weights = neuralNet->GetWeights();

			if (_startWeightGenerator == StartWeightGenerator::UniformDistribution) {
				float factor = 4.0f*(sqrt(6.0/(_visibleStatesCount + _hiddenStatesCount)));
				std::uniform_real_distribution<float> disUniform(-factor, factor);
				for (int i = 0; i < weightsCount; i++) {
					weights[i] = disUniform(gen);
				}
			}
			else if (_startWeightGenerator == StartWeightGenerator::NormalDistribution) {
				float sigma = 2.0f*(sqrt(6.0/(_visibleStatesCount + _hiddenStatesCount)));
				std::normal_distribution<float> disNormal(0, sigma);
				for (int i = 0; i < weightsCount; i++) {
					weights[i] = disNormal(gen);
				}
			}

			if (_startWeightGenerator != StartWeightGenerator::NullDistribution) {
				float *visibleStatesBias = neuralNet->GetVisibleStatesBias();
				for (int i = 0; i < visibleStatesCount; i++) {
					visibleStatesBias[i] = 0.0f;
				}

				float *hiddenStatesBias = neuralNet->GetHiddenStatesBias();
				for (int i = 0; i < hiddenStatesCount; i++) {
					hiddenStatesBias[i] = 0.0f;
				}
			}

			return neuralNet;
		}

		RestrictedBoltzmannMachineBase* RestrictedBoltzmannMachineFactory::InstantiateRbm(RbmType rbmType) {
			RestrictedBoltzmannMachineBase *rbm = 0;
			switch (rbmType) {
				case RbmType::BinaryBinary:
					rbm = new BinaryBinaryRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType::BinaryNrelu:
					//rbm = new BinaryNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType::GaussianBinary:
					rbm = new GaussianBinaryRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType::GaussianNrelu:
					//rbm = new GaussianNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
				case RbmType::ReluNrelu:
					//rbm = new ReluNreluRbm(_visibleStatesCount, _hiddenStatesCount);
					break;
			}
			return rbm;
		}
	}
}