#pragma once

#include "ExportDll.h"
#include "NeuralNet.h"
#include "NeuralNetFactory.h"
#include "RestrictedBoltzmannMachine.h"

namespace NeuralNetNative {
	namespace RestrictedBoltzmannMachine {
		enum RbmType {
			BinaryBinary,
			BinaryNrelu,
			GaussianBinary,
			GaussianNrelu,
			ReluNrelu
		};

		class NEURALNETNATIVE_EXPORT RestrictedBoltzmannMachineFactory : public NeuralNetFactory {
		private:
			int _visibleStatesCount;
			int _hiddenStatesCount;
			StartWeightGenerator _startWeightGenerator;
			RbmType _rbmType;
		public:
			RestrictedBoltzmannMachineFactory(RbmType rbmType, int visibleStatesCount, int hiddenStatesCount, StartWeightGenerator startWeightGenerator);
			~RestrictedBoltzmannMachineFactory(void);
			NeuralNet* CreateNeuralNet(void);
		private:
			RestrictedBoltzmannMachineBase* InstantiateRbm(RbmType rbmType);
		};
	}
}