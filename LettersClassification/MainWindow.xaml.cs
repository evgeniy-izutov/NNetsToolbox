using System;
using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Research.DynamicDataDisplay;
using Microsoft.Research.DynamicDataDisplay.DataSources;
using MathNet.Numerics.Statistics;
using NeuralNet;
using NeuralNet.ActivationFunctions;
using NeuralNet.LeanFactorStrategy;
using NeuralNet.MultyLayerPerceptron;
using NeuralNet.RegularizationFunctions;
using StandardTypes;

namespace LettersClassification {
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        private const string TrainDataFilePath = "..\\..\\Source\\MNIST\\train_data.data";
		private const string TrainLabelFilePath = "..\\..\\Source\\MNIST\\train_label.data";
		private const string TestDataFilePath = "..\\..\\Source\\MNIST\\test_data.data";
		private const string TestLabelFilePath = "..\\..\\Source\\MNIST\\test_label.data";
    	private const string WeightsStatPath = "..\\..\\Source\\WeightsStat\\";
        private const float TrueValue = 1.0f;
        private const float FalseValue = 0.0f;
	    private const int ImageSize = 28;
        private const int HiddenLayer1Size = 500;
		private const int HiddenLayer2Size = 300;
		private const int HiddenLayer3Size = 100;
		private const int HiddenLayer4Size = 50;
        private const int OutputLayerSize = 10;
		private const int NumberPretrainedHiddenLayers = 3;
        private INeuralNet _neuralNet;
        private float[] _neuralNetOutput;
    	private float[] _neuralNetOutputForTest;
        private List<TrainPair> _trainData;
		private List<TrainPair> _testData;
	    private byte[][] _sourceTestImages;
        private INormalizeMethod _normalizeMethod;
        private float[] _coordinateX, _coordinateY;
        private EnumerableDataSource<float> _changedDataSource;
    	private TrainProperties _trainProperties;
		private ObservableDataSource<Point> _sourceTrain;
		private ObservableDataSource<Point> _sourceCrossValidation;
		private ObservableDataSource<Point>[] _preTrainedProgress;
		private ObservableDataSource<Point>[] _preTestedProgress;
		private readonly Thread _workingThread;
    	private Stopwatch _stopWatch;
		private int _preTrainedIter;
		private int[] _neuronsCount;
	    //private RestrictedBoltzmannMachine[] _rbms;

    	public MainWindow() {
            InitializeComponent();
            AllocateMemory();
    		CreateProgressChart();
			CreateChartForShowing();

			LoadTrainData();
            LoadTestData();
			NormalizeData();

            _workingThread = new Thread(CompleteMainProcedure);
			_workingThread.Start();
        }

        private void AllocateMemory() {
			_normalizeMethod = new SigmaComponentAnalysis();
            _coordinateX = new float[OutputLayerSize];
            _coordinateY = new float[OutputLayerSize];
            _neuralNetOutput = new float[OutputLayerSize];
			_neuralNetOutputForTest = new float[OutputLayerSize];
			_sourceTrain = new ObservableDataSource<Point>();
			_sourceCrossValidation = new ObservableDataSource<Point>();
			_stopWatch = new Stopwatch();
			//_rbms = new RestrictedBoltzmannMachine[NumberPretrainedHiddenLayers];
        }

		private void CreateProgressChart() {
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_sourceTrain) {
                Stroke = Brushes.Blue,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_sourceCrossValidation) {
                Stroke = Brushes.Red,
                StrokeThickness = 1.2
            });
			
			_preTrainedProgress = new ObservableDataSource<Point>[4];
			_preTestedProgress = new ObservableDataSource<Point>[4];
			for (var i = 0; i < 4; i++) {
				_preTrainedProgress[i] = new ObservableDataSource<Point>();
				_preTestedProgress[i] = new ObservableDataSource<Point>();
			}

            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTrainedProgress[0]) {
                Stroke = Brushes.Blue,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTrainedProgress[1]) {
                Stroke = Brushes.Blue,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTrainedProgress[2]) {
                Stroke = Brushes.Blue,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTrainedProgress[3]) {
                Stroke = Brushes.Blue,
                StrokeThickness = 1.2
            });
            
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTestedProgress[0]) {
                Stroke = Brushes.Red,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTestedProgress[1]) {
                Stroke = Brushes.Red,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTestedProgress[2]) {
                Stroke = Brushes.Red,
                StrokeThickness = 1.2
            });
            ChartPlotterTrainProgress.Children.Add(new LineGraph(_preTestedProgress[3]) {
                Stroke = Brushes.Red,
                StrokeThickness = 1.2
            });
		}

		private void LoadTrainData () {
			if (File.Exists(TrainDataFilePath) && File.Exists(TrainLabelFilePath)) {
				var dataLoader = new MnistDataLoader(TrainDataFilePath, TrainLabelFilePath, false, TrueValue, FalseValue);
				_trainData = dataLoader.LoadData(null, null);
			}
        }

        private void LoadTestData() {
            if (File.Exists(TestDataFilePath) && File.Exists(TestLabelFilePath)) {
				var dataLoader = new MnistDataLoader(TestDataFilePath, TestLabelFilePath, false, TrueValue, FalseValue);
				_testData = dataLoader.LoadData(null, null);

				_sourceTestImages = new byte[_testData.Count][];
				for (var i = 0; i < _testData.Count; i++) {
					ListBoxTrainExample.Items.Add("Test example " + i);

					var input = _testData[i].Input;
					var exampleImage = new byte[ImageSize*ImageSize];
					for (var j = 0; j < ImageSize*ImageSize; j++) {
						exampleImage[j] = Convert.ToByte(255f - input[j]);
					}
					_sourceTestImages[i] = exampleImage;
				}
            }
        }

	    private void NormalizeData() {
		    _normalizeMethod.CollectStatistics(_trainData);
            _normalizeMethod.NormalizeSet(_trainData, false, true);

			//_normalizeMethod.CollectStatistics(_testData);
            _normalizeMethod.NormalizeSet(_testData , false, true);
	    }

        private void CreateChartForShowing() {
            for (var i = 0; i < OutputLayerSize; i++) {
                _coordinateX[i] = i;
                _coordinateY[i] = 0.0f;
            }

            var xSrc = new EnumerableDataSource<float>(_coordinateX);
            xSrc.SetXMapping(x => x);
            _changedDataSource = new EnumerableDataSource<float>(_coordinateY);
            _changedDataSource.SetYMapping(y => y);

            ChartPlotterPredict.Children.Add(new LineGraph(new CompositeDataSource(xSrc, _changedDataSource)) {
                Stroke = Brushes.Goldenrod,
                StrokeThickness = 2
            });
            ChartPlotterPredict.FitToView();
        }

        private void CreateNeuronNet() {
            var inputLayerSzie = _normalizeMethod.InputVectorSize;
			_neuronsCount = new[]{inputLayerSzie, HiddenLayer1Size, HiddenLayer2Size, OutputLayerSize};
			var hiddenLayersFunction = new HyperbolicTangens(1.0f, 1.0f);
        	var outputLayerFunction = new Softmax();
	        var weightGenerator = new ImprovedBestActiveZone(Distribution.Normal, _trainData);
			//var weightGenerator = new BestActiveZone(Distribution.Normal);
			var neuronFactory = new Factory(_neuronsCount, hiddenLayersFunction, outputLayerFunction, weightGenerator);
            _neuralNet = neuronFactory.CreateNeuralNet();

			//var iterCounts = new[] {400, 400, 500, 600};

			//var perceptronLayers = ((MultyLayerPerceptron) _neuralNet).Layers;
			//var pretrainedData = BuildStartPretrainedData(_trainData);
			//var pretestedData = BuildStartPretrainedData(_testData);
			//for (_preTrainedIter = 0; _preTrainedIter < NumberPretrainedHiddenLayers; _preTrainedIter++) {
			//	var curLayer = perceptronLayers[_preTrainedIter];

			//	var rbmFactory = _preTrainedIter == 0 ? 
			//		new RestrictedBoltzmannMachineFactory(RbmType.BinaryBinary, _neuronsCount[_preTrainedIter], _neuronsCount[_preTrainedIter + 1], DistributionType.Normal) : 
			//		new RestrictedBoltzmannMachineFactory(RbmType.BinaryBinary, _neuronsCount[_preTrainedIter], _neuronsCount[_preTrainedIter + 1], DistributionType.Normal);
				
			//	_rbms[_preTrainedIter] = (RestrictedBoltzmannMachine) rbmFactory.CreateNeuralNet();

			//	var trainProperties = new TrainProperties {
			//		Epsilon = 0.01f,
			//		MaxIterationCount = iterCounts[_preTrainedIter],
			//		Metrics = new SquaredEuclidianDistance(),
			//		PackageSize = 10,
			//		CvLimit = 10.0f,
			//      SkipCvLimitFirstIterations = 10,
			//   	CvSlidingFactor = 0.5f,
			//		BaseLearnSpeed = 0.01f,
			//		SpeedBonus = 0.05f,
			//		SpeedPenalty = 0.95f,
			//		SpeedUpBorder = 100,
			//		SpeedLowBorder = 0.001f,
			//		AverageLearnFactor = 0.6f,
			//		Momentum = 0.9f,
			//		Regularization = new EliminationRegularization(1.0f, 0.5f)
			//	};

			//	//var trainMethod = new ContrastiveDivergence(pretrainedData, pretestedData, 10);
			//	var trainMethod = new NeuralNetNativeWrapper.RestrictedBoltzmannMachineNativeWrapper.ContrastiveDivergenceNative(pretrainedData, pretestedData, 10);
			//	trainMethod.InitilazeMethod(_rbms[_preTrainedIter], trainProperties);
			//	trainMethod.IterationCompleted += PreTrainingIterationCompleted;

			//	trainMethod.Start();

			//	pretrainedData = BuildNextPretrainedData(_rbms[_preTrainedIter], pretrainedData, _neuronsCount[_preTrainedIter + 1], true, 1000);
			//	pretestedData = BuildNextPretrainedData(_rbms[_preTrainedIter], pretestedData, _neuronsCount[_preTrainedIter + 1], true, 1000);
			//	SetWeights(_rbms[_preTrainedIter].Weights, _rbms[_preTrainedIter].HiddenStatesBias, curLayer.GetWeights()[0], curLayer.GetBias());
			//}
        }

		private static IList<TrainSingle> BuildStartPretrainedData(IList<TrainPair> data) {
			var retList = new List<TrainSingle>(data.Count);
			foreach (var example in data) {
				retList.Add(new TrainSingle(example));
			}
			return retList;
		}

		//private static IList<TrainSingle> BuildNextPretrainedData(RestrictedBoltzmannMachine rbmNet, IList<TrainSingle> sourceData, 
		//	int outputSize, bool isSamplingOutput) {

		//	var retList = new List<TrainSingle>(sourceData.Count);
		//	for (var i = 0; i < sourceData.Count; i++) {
		//		var output = new float[outputSize];
		//		rbmNet.CalculateHiddenStates(sourceData[i].Input, output, isSamplingOutput);
		//		retList.Add(new TrainSingle(output));
		//	}
		//	return retList;
		//}

		//private static void SetWeights(float[] sourceWeights, float[] sourceBias, float[] targetWeights, float[] targetBias) {
		//	for (var i = 0; i < targetWeights.Length; i++) {
		//		targetWeights[i] = sourceWeights[i];
		//	}
		//	for (var i = 0; i < targetBias.Length; i++) {
		//		targetBias[i] = sourceBias[i];
		//	}
		//}

		private void CompleteMainProcedure() {
			CreateNeuronNet();
			TrainNeuralNet();

			TestNeuralNet();
			CalculateWeightsStat();
		}
        
        private void TrainNeuralNet () {
        	_trainProperties = new TrainProperties {
        		Epsilon = 0.0001f,
        		MaxIterationCount = 25,
				CvLimit = 0.01f,
				SkipCvLimitFirstIterations = 10,
				CvSlidingFactor = 0.5f,
        		Metrics = new CrossEntropyForSoftmax(),
        		PackageSize = 50,
				BaseLearnSpeed = 0.01f,
				SpeedBonus = 0.001f,
				SpeedPenalty = 0.999f,
				SpeedUpBorder = float.MaxValue,
				SpeedLowBorder = float.MinValue,
				LearnFactorStrategy = new ReverseFactor(),
				AverageLearnFactor = 0.7f,
				Momentum = 0.99f,
        		Regularization = new Elimination(0.001f, 1.2f)
				//Regularization = new L2Regularization(0.00001f)
				//Regularization = new NoRegularization()
        	};

	        //_trainData = BuildPretranedData(_trainData);
			//_testData = BuildPretranedData(_testData);
			
			//var trainMethod = new BackPropagationAlgorithm(_trainData, _testData);
			var trainMethod = new NeuralNetNativeWrapper.MultyLayerPerceptronNativeWrapper.BackPropagationAlgorithmNative(_trainData, _testData);
            trainMethod.InitilazeMethod(_neuralNet, _trainProperties);
            trainMethod.IterationCompleted += TrainingIterationCompleted;

            _stopWatch.Reset();
            _stopWatch.Start();
            trainMethod.Start();
            _stopWatch.Stop();
        }

		//private List<TrainPair> BuildPretranedData(List<TrainPair> data) {
		//	var result = new List<TrainPair>(data.Count);
		    
		//	var singleData = BuildStartPretrainedData(data);
		//	for (var i = 0; i < NumberPretrainedHiddenLayers - 1; i++) {
		//		singleData = BuildNextPretrainedData(_rbms[i], singleData, _neuronsCount[i + 1], true);
		//	}
		//	singleData = BuildNextPretrainedData(_rbms[NumberPretrainedHiddenLayers - 1], singleData,
		//		_neuronsCount[NumberPretrainedHiddenLayers], false);

		//	for (var i = 0; i < data.Count; i++) {
		//		var newPair = new TrainPair(singleData[i].Input, data[i].Output, singleData[i].MissedInputIndexes,
		//			data[i].MissedOutputIndexes);
		//		newPair.Id = data[i].Id;
		//		newPair.Weight = data[i].Weight;
		//		result.Add(newPair);
		//	}

		//	return result;
		//} 

        private void TrainingIterationCompleted (object sender, IterationCompletedEventArgs e) {
            var pointTrain = new Point(e.IterationNum, e.IterationValue);
            _sourceTrain.AppendAsync(Dispatcher, pointTrain);

            if (!float.IsNaN(e.AddedIterationValue)) {
		        var pointCrossValidation = new Point(e.IterationNum, e.AddedIterationValue);
				_sourceCrossValidation.AppendAsync(Dispatcher, pointCrossValidation);
	        }
        }

		//private void PreTrainingIterationCompleted (object sender, IterationCompletedEventArgs e) {
		//	var pointLearning = new Point(e.IterationNum, e.IterationValue);
		//	_preTrainedProgress[_preTrainedIter].AppendAsync(Dispatcher, pointLearning);

		//	if (!float.IsNaN(e.AddedIterationValue)) {
		//		var pointTesting = new Point(e.IterationNum, e.AddedIterationValue);
		//		_preTestedProgress[_preTrainedIter].AppendAsync(Dispatcher, pointTesting);
		//	}
		//}

		private void ListBoxLearningPairsSelectionChanged(object sender, SelectionChangedEventArgs e) {
            var imageIndex = ListBoxTrainExample.SelectedIndex;
            ShowPicture(_sourceTestImages[imageIndex]);

			_neuralNet.Predict(_testData[imageIndex].Input, _neuralNetOutput);
            ShowChart(_neuralNetOutput);
        }

	    private void ListBoxTestPairsSelectionChanged(object sender, SelectionChangedEventArgs e) {
            var imageIndexName = (string) ListBoxWrongExample.SelectedValue;
		    var imageIndex = Convert.ToInt32(imageIndexName.Substring(14));
			ShowPicture(_sourceTestImages[imageIndex]);

            _neuralNet.Predict(_testData[imageIndex].Input, _neuralNetOutput);
            ShowChart(_neuralNetOutput);
        }

		private void ShowPicture (byte[] imageData) {
			const int stride = ImageSize * ((8 + 7) / 8);
            var bitmapImage = BitmapSource.Create(ImageSize, ImageSize, 96d, 96d, PixelFormats.Gray8, null, imageData, stride);

			image.Source = bitmapImage;
        }

        private void ShowChart (float[] outputVector) {
            for (var i = 0; i < OutputLayerSize; i++) {
                _coordinateY[i] = outputVector[i];
            }
            _changedDataSource.RaiseDataChanged();
        }

		private void CalculateWeightsStat() {
			const int barCount = 10;
			var outputTotalStat = new StreamWriter(WeightsStatPath + "WeightsStat.txt");
			var layers = ((MultyLayerPerceptron) _neuralNet).Layers;
			for (var i = 0; i < layers.Length; i++) {
				var weights = layers[i].GetWeights()[0];

				var outputCurWeights = new StreamWriter(WeightsStatPath + "WeightsOfLayer_" + i + ".txt");
				for (var j = 0; j < weights.Length; j++) {
					outputCurWeights.WriteLine(weights[j]);
				}
				outputCurWeights.Close();

				var copyWeights = new double[weights.Length];
				weights.CopyTo(copyWeights, 0);
				var histrogramm = new Histogram(copyWeights, barCount);
				outputTotalStat.WriteLine("Layer " + (i + 1).ToString());
				for (var j = 0; j < histrogramm.BucketCount; j++) {
					var bar = histrogramm[j];
					outputTotalStat.WriteLine(bar.ToString());
				}
				outputTotalStat.WriteLine();
			}
			outputTotalStat.Close();
		}

		private void TestNeuralNet() {
			Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => {
				StatusBarItemTime.Content = "Time: " + _stopWatch.Elapsed.TotalSeconds;
				StatusBarItemCrossValidationError.Content = "Train error: " + GetErrorOnSet(_trainData);
				StatusBarItemTestError.Content = "Test error: " + GetErrorOnSet(_testData);
				StatusBarItemTrainPercent.Content = "Error on train: " + (1f - GetPercentOnSet(_trainData, false))*100 + "%";
				StatusBarItemTestPercent.Content = "Error on test: " + (1f - GetPercentOnSet(_testData, true))*100 + "%";
			}));
		}

		private float GetErrorOnSet(List<TrainPair> list) {
			var error = 0.0f;
			foreach (var pair in list) {
				_neuralNet.Predict(pair.Input, _neuralNetOutputForTest);
				error += _trainProperties.Metrics.Calculate(pair.Output, _neuralNetOutputForTest);
			}
			return error/list.Count;
		}

		private float GetPercentOnSet(List<TrainPair> list, bool isAddToList) {
			var rightAnswers = 0.0f;
			foreach (var pair in list) {
				_neuralNet.Predict(pair.Input, _neuralNetOutputForTest);
				var isRightAnswer = IsRightAnswer(_neuralNetOutputForTest, pair.Output);
				rightAnswers += isRightAnswer;

				if (isRightAnswer == 0 && isAddToList) {
					Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => {
						ListBoxWrongExample.Items.Add("Wrong example " + pair.Id);
					}));
				}
			}
			return rightAnswers/list.Count;
		}

		private static int IsRightAnswer(float[] neuronNetOutput, float[] realOutput) {
			var size = neuronNetOutput.Length;
			var winnerPos = 0;
			var maxValue = neuronNetOutput[0];
			for (var i = 1; i < size; i++) {
                var value = neuronNetOutput[i];
				if (value > maxValue) {
					maxValue = value;
					winnerPos = i;
				}
            }
			return Math.Abs(realOutput[winnerPos] - TrueValue) < float.Epsilon ? 1 : 0;
		}

		private void WindowClosing(object sender,System.ComponentModel.CancelEventArgs e) {
			_workingThread.Abort();
		}
    }
}
