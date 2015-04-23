#include "TrainMethodNative.h"
#include "HalfSquaredEuclidianDistance.h"
#include "CrossEntropy.h"
#include "Loglikelihood.h"
#include "HammingDistance.h"
#include "L1Regularization.h"
#include "L2Regularization.h"
#include "EliminationRegularization.h"
#include "NoRegularization.h"
#include "RestrictedBoltzmannMachineFactory.h"
#include "Callback.h"
#include "malloc.h"
#include "ConstantFactor.h"
#include "ReverseFactor.h"
#include "SqrtReverseFactor.h"
#include "LinearFactor.h"

using namespace NeuralNet::RegularizationFunctions;
using namespace StandardTypes::FactorStrategy;

namespace NeuralNetNativeWrapper {
    generic<class T> where T:TrainData
	void TrainMethodNative<T>::InitilazeMethod(INeuralNet^ neuralNet, ITrainProperties<T>^ trainProperties) {
		DeleteNativeNeuralNet();
		DeleteNativeProperties();
		_properties = trainProperties;
		CreateNativeTrainProperties(trainProperties);
		CreateNativeNeuralNet(neuralNet);
		InitilazeNativeAlgorithm();
	}

    generic<class T> where T:TrainData
	ITrainProperties<T>^ TrainMethodNative<T>::Properties::get(void) {
		return _properties;
	}

    generic<class T> where T:TrainData
	void TrainMethodNative<T>::CreateNativeTrainProperties(ITrainProperties<T>^ trainProperties) {
		_nativeTrainProperties = new NeuralNetNative::TrainProperties();
		_nativeTrainProperties->Epsilon = trainProperties->Epsilon;
		_nativeTrainProperties->MaxIterationCount = trainProperties->MaxIterationCount;
		_nativeTrainProperties->PackageSize = trainProperties->PackageSize;
		_nativeTrainProperties->CvLimit = trainProperties->CvLimit;
        _nativeTrainProperties->SkipCvLimitFirstIterations = trainProperties->SkipCvLimitFirstIterations;
        _nativeTrainProperties->CvSlidingFactor = trainProperties->CvSlidingFactor;
		_nativeTrainProperties->BaseLearnSpeed = trainProperties->BaseLearnSpeed;
		_nativeTrainProperties->SpeedBonus = trainProperties->SpeedBonus;
		_nativeTrainProperties->SpeedPenalty = trainProperties->SpeedPenalty;
		_nativeTrainProperties->SpeedUpBorder = trainProperties->SpeedUpBorder;
		_nativeTrainProperties->SpeedLowBorder = trainProperties->SpeedLowBorder;
		_nativeTrainProperties->AverageLearnFactor = trainProperties->AverageLearnFactor;
		_nativeTrainProperties->Momentum = trainProperties->Momentum;

		StandardTypesNative::Metrics *nativeMetrics;
		IMetrics^ metrics = trainProperties->Metrics;
		if (dynamic_cast<HammingDistance^>(metrics) != nullptr) {
			nativeMetrics = new StandardTypesNative::HammingDistance();
		}
		else if (dynamic_cast<SquaredEuclidianDistance^>(metrics) != nullptr) {
			nativeMetrics = new StandardTypesNative::HalfSquaredEuclidianDistance();
		}
		else if (dynamic_cast<CrossEntropyForSoftmax^>(metrics) != nullptr) {
			nativeMetrics = new StandardTypesNative::CrossEntropyForSoftmax();
		}
		else {
			nativeMetrics = new StandardTypesNative::LoglikelihoodForSoftmax();
		}
		_nativeTrainProperties->Metrics = nativeMetrics;

		NeuralNetNative::Regularization *nativeRegularization;
		Regularization^ regularization = trainProperties->Regularization;
		if (dynamic_cast<L1^>(regularization) != nullptr) {
			nativeRegularization = new NeuralNetNative::L1Regularization(regularization->Factor);
		}
		else if (dynamic_cast<L2^>(regularization) != nullptr) {
			nativeRegularization = new NeuralNetNative::L2Regularization(regularization->Factor);
		}
		else if (dynamic_cast<Elimination^>(regularization) != nullptr) {
			nativeRegularization = new NeuralNetNative::EliminationRegularization(regularization->Factor,
                dynamic_cast<Elimination^>(regularization)->Alpha);
		}
		else {
			nativeRegularization = new NeuralNetNative::NoRegularization();
		}
		_nativeTrainProperties->Regularization = nativeRegularization;

		NeuralNetNative::LearnFactorStrategy *nativeLearnFactorStrategy;
		IFactorStrategy^ learnFactorStrategy = trainProperties->LearnFactorStrategy;
		if (dynamic_cast<ConstantFactor^>(learnFactorStrategy) != nullptr) {
			nativeLearnFactorStrategy = new NeuralNetNative::ConstantFactor(dynamic_cast<ConstantFactor^>(
                learnFactorStrategy)->ConstantValue);
		}
		else if (dynamic_cast<ReverseFactor^>(learnFactorStrategy) != nullptr) {
			nativeLearnFactorStrategy = new NeuralNetNative::ReverseFactor();
		}
		else if (dynamic_cast<SqrtReverseFactor^>(learnFactorStrategy) != nullptr) {
			nativeLearnFactorStrategy = new NeuralNetNative::SqrtReverseFactor();
		}
		else if (dynamic_cast<LinearFactor^>(learnFactorStrategy) != nullptr) {
			LinearFactor^ factor = dynamic_cast<LinearFactor^>(learnFactorStrategy);
			nativeLearnFactorStrategy = new NeuralNetNative::LinearFactor(factor->A, factor->B);
		}
		_nativeTrainProperties->FactorStrategy = nativeLearnFactorStrategy;

		NeuralNetNative::LearnFactorStrategy *nativeAddedLearnFactorStrategy;
		IFactorStrategy^ addedLearnFactorStrategy = trainProperties->AddedLearnFactorStrategy;
		if (dynamic_cast<ConstantFactor^>(addedLearnFactorStrategy) != nullptr) {
			nativeAddedLearnFactorStrategy = new NeuralNetNative::ConstantFactor(dynamic_cast<ConstantFactor^>(
                addedLearnFactorStrategy)->ConstantValue);
		}
		else if (dynamic_cast<ReverseFactor^>(addedLearnFactorStrategy) != nullptr) {
			nativeAddedLearnFactorStrategy = new NeuralNetNative::ReverseFactor();
		}
		else if (dynamic_cast<SqrtReverseFactor^>(addedLearnFactorStrategy) != nullptr) {
			nativeAddedLearnFactorStrategy = new NeuralNetNative::SqrtReverseFactor();
		}
		else if (dynamic_cast<LinearFactor^>(addedLearnFactorStrategy) != nullptr) {
			LinearFactor^ factor = dynamic_cast<LinearFactor^>(addedLearnFactorStrategy);
			nativeAddedLearnFactorStrategy = new NeuralNetNative::LinearFactor(factor->A, factor->B);
		}
		_nativeTrainProperties->AddedFactorStrategy = nativeAddedLearnFactorStrategy;
	}

    generic<class T> where T:TrainData
	void TrainMethodNative<T>::DeleteNativeProperties(void) {
		if (_nativeTrainProperties != 0) {
			delete _nativeTrainProperties->Metrics;
			delete _nativeTrainProperties->Regularization;
			delete _nativeTrainProperties;
			_nativeTrainProperties = 0;
		}
	}

    generic<class T> where T:TrainData
	void TrainMethodNative<T>::IterationCompletedHandler(int iterationNum, float iterationValue, float addedIterationValue) {
		IterationCompleted(this, gcnew IterationCompletedEventArgs(iterationNum, iterationValue, addedIterationValue));
	}

    generic<class T> where T:TrainData
	void TrainMethodNative<T>::IterativeProcessFinishedHandler(int iterationCount) {
		IterativeProcessFinished(this, gcnew IterativeProcessFinishedEventArgs(iterationCount));
	}
}
