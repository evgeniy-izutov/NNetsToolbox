using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Research.DynamicDataDisplay;
using Microsoft.Research.DynamicDataDisplay.DataSources;
using NeuralNet;
using NeuralNet.ClassificationRbm;
using NeuralNet.RegularizationFunctions;
using StandardTypes;
using StandardTypes.FactorStrategy;
using StandardTypes.SetWeights;

namespace RbmLettersClassification {
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
	    private const int InputLayerSize = ImageSize*ImageSize;
	    private const int HiddenLayerSize = 49;
        private const int OutputLayerSize = 10;
        private ClassificationRbm _neuralNet;
        private float[] _neuralNetOutput;
    	private float[] _neuralNetOutputForTest;
        private List<TrainPair> _trainData;
		private List<TrainPair> _testData;
	    private byte[][] _sourceTestImages;
        private float[] _coordinateX, _coordinateY;
        private EnumerableDataSource<float> _changedDataSource;
    	private TrainProperties<TrainPair> _trainProperties;
		private ObservableDataSource<Point> _sourceTrain;
		private ObservableDataSource<Point> _sourceCrossValidation;
		private readonly Thread _workingThread;
    	private Stopwatch _stopWatch;

    	public MainWindow() {
            InitializeComponent();
            AllocateMemory();
    		CreateProgressChart();
			CreateChartForShowing();

			LoadTrainData();
            LoadTestData();

            _workingThread = new Thread(CompleteMainProcedure);
			_workingThread.Start();
        }

        private void AllocateMemory() {
            _coordinateX = new float[OutputLayerSize];
            _coordinateY = new float[OutputLayerSize];
            _neuralNetOutput = new float[OutputLayerSize];
			_neuralNetOutputForTest = new float[OutputLayerSize];
			_sourceTrain = new ObservableDataSource<Point>();
			_sourceCrossValidation = new ObservableDataSource<Point>();
			_stopWatch = new Stopwatch();
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
		}

		private void LoadTrainData () {
			if (File.Exists(TrainDataFilePath) && File.Exists(TrainLabelFilePath)) {
				var dataLoader = new MnistDataLoader(TrainDataFilePath, TrainLabelFilePath, true, TrueValue, FalseValue);
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

				dataLoader = new MnistDataLoader(TestDataFilePath, TestLabelFilePath, true, TrueValue, FalseValue);
				_testData = dataLoader.LoadData(null, null);
			}
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

        private void CreateNeuronNet(float[] inputProbability, float[] labelProbability) {
			var neuronFactory = new Factory(InputLayerSize, HiddenLayerSize, OutputLayerSize,
				DistributionType.Uniform, inputProbability, labelProbability);
            _neuralNet = neuronFactory.CreateNeuralNet() as ClassificationRbm;
        }

		private void CompleteMainProcedure() {
			float[] visibleUnitsProbability;
			float[] priorClassDistribution;
			CalculateVisibleUnitsProbability(out visibleUnitsProbability, out priorClassDistribution);
			
			CreateNeuronNet(null, null);
			TrainNeuralNet();

			TestNeuralNet();
			CalculateWeightsStat();
		}
        
		private void CalculateVisibleUnitsProbability(out float[] inputProbability, out float[] labelProbability) {
		    var factor = 1.0f/_trainData.Count;
			
			inputProbability = new float[_trainData[0].InputLength];
			labelProbability = new float[_trainData[0].OutputLength];
		    
		    foreach (var example in _trainData) {
			    var input = example.Input;
			    for (var i = 0; i < input.Length; i++) {
				    inputProbability[i] += ((Math.Abs(input[i] - TrueValue) < float.Epsilon) ? 1f : 0f)*factor;
			    }

				var output = example.Output;
			    for (var i = 0; i < output.Length; i++) {
				    labelProbability[i] += ((Math.Abs(output[i] - TrueValue) < float.Epsilon) ? 1f : 0f)*factor;
			    }
		    }
	    }

        private void TrainNeuralNet () {
        	_trainProperties = new TrainProperties<TrainPair> {
        		Epsilon = 0.0001f,
        		MaxIterationCount = 25,
				CvLimit = 0.01f,
				SkipCvLimitFirstIterations = 5,
				CvSlidingFactor = 0.8f,
        		Metrics = new CrossEntropyForSoftmax(),
        		PackageSize = 50,
				BaseLearnSpeed = 0.01f,
				SpeedBonus = 0.001f,
				SpeedPenalty = 0.999f,
				SpeedUpBorder = float.MaxValue,
				SpeedLowBorder = float.MinValue,
				LearnFactorStrategy = new SqrtReverseFactor(),
				AverageLearnFactor = 0.6f,
				Momentum = 0.96f,

				SetWeightsAdaptation = new ExponentialWeights<TrainPair>(new SqrtReverseFactor(0.1f)),

        		//Regularization = new Elimination(0.001f, 1.2f)
				Regularization = new L1(0.0001f)
				//Regularization = new NoRegularization()
        	};
			
			//var trainMethod = new GenerativeTrainMethod(_trainData, _testData, 1);
			//var trainMethod = new DiscriminativeTrainMethod(_trainData, _testData);
	        var trainMethod = new HybridTrainMethod(_trainData, _testData, 1, 0.01f);
            trainMethod.InitilazeMethod(_neuralNet, _trainProperties);
            trainMethod.IterationCompleted += TrainingIterationCompleted;

            _stopWatch.Reset();
            _stopWatch.Start();
            trainMethod.Start();
            _stopWatch.Stop();
        }

        private void TrainingIterationCompleted(object sender, IterationCompletedEventArgs e) {
            var pointTrain = new Point(e.IterationNum, e.IterationValue);
            _sourceTrain.AppendAsync(Dispatcher, pointTrain);

            if (!float.IsNaN(e.AddedIterationValue)) {
		        var pointCrossValidation = new Point(e.IterationNum, e.AddedIterationValue);
				_sourceCrossValidation.AppendAsync(Dispatcher, pointCrossValidation);
	        }
        }

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
			var visibleWeights = _neuralNet.VisibleStatesWeights;
			var weightDoublePrecition = new double[visibleWeights.Length];
			var outputVisibleWeights = new StreamWriter(WeightsStatPath + "VisibleWeightsOfClassRBM.txt");
			for (var j = 0; j < visibleWeights.Length; j++) {
				outputVisibleWeights.WriteLine(visibleWeights[j]);
				weightDoublePrecition[j] = visibleWeights[j];
			}
			outputVisibleWeights.Close();

			var labelWeights = _neuralNet.LabelsWeights;
			weightDoublePrecition = new double[labelWeights.Length];
			var outputLabelWeights = new StreamWriter(WeightsStatPath + "LabelWeightsOfClassRBM.txt");
			for (var j = 0; j < labelWeights.Length; j++) {
				outputLabelWeights.WriteLine(labelWeights[j]);
				weightDoublePrecition[j] = labelWeights[j];
			}
			outputLabelWeights.Close();

			var visibleBias = _neuralNet.VisibleStatesBias;
			var outputVisibleBias = new StreamWriter(WeightsStatPath + "VisibleBiasOfClassRBM.txt");
			for (var i = 0; i < visibleBias.Length; i++) {
				outputVisibleBias.WriteLine(visibleBias[i]);
			}
			outputVisibleBias.Close();

			var labelBias = _neuralNet.LabelsBias;
			var outputLabelBias = new StreamWriter(WeightsStatPath + "LabelBiasOfClassRBM.txt");
			for (var i = 0; i < labelBias.Length; i++) {
				outputLabelBias.WriteLine(labelBias[i]);
			}
			outputLabelBias.Close();

			var hiddenBias = _neuralNet.HiddenStatesBias;
			var outputHiddenBias = new StreamWriter(WeightsStatPath + "HiddenBiasOfClassRBM.txt");
			for (var i = 0; i < hiddenBias.Length; i++) {
				outputHiddenBias.WriteLine(hiddenBias[i]);
			}
			outputHiddenBias.Close();

			const int vectorSize = ImageSize*ImageSize;
			var blocksCount = (int) Math.Sqrt(HiddenLayerSize);
			var pixels = new byte[visibleWeights.Length];

			for (var i = 0; i < blocksCount; i++) {
				for (var j = 0; j < blocksCount; j++) {
					var index = i*blocksCount + j;

					var localMax = float.MinValue;
					var localMin = float.MaxValue;
					for (var k = 0; k < vectorSize; k++) {
						var weight = visibleWeights[index*vectorSize + k];
						if (weight > localMax) {
							localMax = weight;
						}
						else if (weight < localMin) {
							localMin = weight;
						}
					}

					for (var m = 0; m < ImageSize; m++) {
						for (var n = 0; n < ImageSize; n++) {
							var weight = visibleWeights[index*vectorSize + m*ImageSize + n];
							pixels[i*blocksCount*vectorSize + j*ImageSize + m*blocksCount*ImageSize + n ] = 
								(byte) ((weight - localMin)*255.0/(localMax - localMin));
						}
					}
				}
			}
			var bitmapImage = BitmapSource.Create(ImageSize*blocksCount, ImageSize*blocksCount,
				96d, 96d, PixelFormats.Gray8, null, pixels, ImageSize*blocksCount);
			var stream = new FileStream(WeightsStatPath + "weightsForClassRBM.png", FileMode.Create);
			var encoder = new PngBitmapEncoder();
			encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
			encoder.Save(stream);
			stream.Close();
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
