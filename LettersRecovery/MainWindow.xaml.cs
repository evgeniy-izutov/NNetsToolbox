using System;
using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Research.DynamicDataDisplay;
using Microsoft.Research.DynamicDataDisplay.DataSources;
using MathNet.Numerics.Statistics;
using NeuralNet;
using NeuralNet.RestrictedBoltzmannMachine;
using StandardTypes;
using Point = System.Windows.Point;
using NativeWrapper = NeuralNetNativeWrapper.RestrictedBoltzmannMachineNativeWrapper;

namespace LettersRecovery {
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow {
        private const string TrainDataFilePath = "..\\..\\Source\\MNIST\\train_data.data";
		private const string TrainLabelFilePath = "..\\..\\Source\\MNIST\\train_label.data";
		private const string TestDataFilePath = "..\\..\\Source\\MNIST\\test_data.data";
		private const string TestLabelFilePath = "..\\..\\Source\\MNIST\\test_label.data";
		private const string WeightsStatPath = "..\\..\\Source\\WeightsStat\\";
    	private const int ImageSize = 28;
        private const float TrueValue = 1.0f;
        private const float FalseValue = 0.0f;
        private const int HiddenStatesCount = 100;
	    private const int VisibleStatesCount = ImageSize*ImageSize;
	    private RestrictedBoltzmannMachine _neuralNet;
        private float[] _neuralNetOutput;
        private List<TrainSingle> _trainData;
		private List<TrainSingle> _testData;
    	private TrainProperties _trainProperties;
		private ObservableDataSource<Point> _sourceTrain;
		private ObservableDataSource<Point> _sourceTest;
		private readonly Thread _workingThread;
    	private Stopwatch _stopWatch;
	    private INormalizeMethod _normalizeMethod;
		private byte[][] _sourceTestImages;

    	public MainWindow() {
            InitializeComponent();

			LoadTrainData();
            LoadTestData();
			//NormalizeData();

            AllocateMemory();
    		CreateProgressChart();

            _workingThread = new Thread(CompleteLearningOfNeuronNet);
			_workingThread.Start();
    	}

        private void AllocateMemory() {
            _neuralNetOutput = new float[VisibleStatesCount];
			_sourceTrain = new ObservableDataSource<Point>();
			_sourceTest = new ObservableDataSource<Point>();
			_stopWatch = new Stopwatch();
        }

		private void CreateProgressChart() {
            chartPlotterProgress.Children.Add(new LineGraph(_sourceTrain) {
		        Stroke = Brushes.Blue,
                StrokeThickness = 1.2
		    });

            chartPlotterProgress.Children.Add(new LineGraph(_sourceTest) {
		        Stroke = Brushes.Red,
                StrokeThickness = 1.2
		    });
		}

		private void LoadTrainData () {
            if (File.Exists(TrainDataFilePath) && File.Exists(TrainLabelFilePath)) {
				var dataLoader = new MnistDataLoader(TrainDataFilePath, TrainLabelFilePath, true, TrueValue, FalseValue);
				_trainData = dataLoader.LoadData(null);
			}
        }

        private void LoadTestData() {
            if (File.Exists(TestDataFilePath) && File.Exists(TestLabelFilePath)) {
				var dataLoader = new MnistDataLoader(TestDataFilePath, TestLabelFilePath, false, TrueValue, FalseValue);
				_testData = dataLoader.LoadData(null);

				_sourceTestImages = new byte[_testData.Count][];
				for (var i = 0; i < _testData.Count; i++) {
					listBoxTestPairs.Items.Add("Test example " + i);

					var input = _testData[i].Input;
					var exampleImage = new byte[ImageSize*ImageSize];
					for (var j = 0; j < ImageSize*ImageSize; j++) {
						exampleImage[j] = Convert.ToByte(255f - input[j]);
					}
					_sourceTestImages[i] = exampleImage;
				}

				//for binary input variant only
				dataLoader = new MnistDataLoader(TestDataFilePath, TestLabelFilePath, true, TrueValue, FalseValue);
				_testData = dataLoader.LoadData(null);
			}
        }

	    private void NormalizeData() {
			_normalizeMethod = new SigmaComponentAnalysis();
			_normalizeMethod.CollectStatistics(_trainData);
			_normalizeMethod.NormalizeSet(_trainData, true);
			_normalizeMethod.NormalizeSet(_testData, true);
	    }

        private static float[] ConvertPictureToArray (string filePath) {
            var fullFilePath = Path.GetFullPath(filePath);
            var pixels = GetPixelsArray(new BitmapImage(new Uri(fullFilePath)));
            var inputVector = GetInputVector(pixels);
            return inputVector;
        }

        private static byte[] GetPixelsArray (BitmapSource bitmapImage) {
            var height = bitmapImage.PixelHeight;
            var width  = bitmapImage.PixelWidth;
            var stride = width * ((bitmapImage.Format.BitsPerPixel + 7) / 8);

            var pixels = new byte[height * stride];
            bitmapImage.CopyPixels(pixels, stride, 0);

            return pixels;
        }

        private static float[] GetInputVector (IList<byte> pixels) {
            var length = pixels.Count;
            var inputVector = new float[length/4];
        	var j = 0;
            for (var i = 0; i < length; i += 4) {
                var value = TrueValue;
                if (pixels[i] > 0 ) {
                    value = FalseValue;
                }
                inputVector[j++] = value;
            }
            return inputVector;
        }

        private void CreateNeuronNet(float[] visibleUnitsProbability) {
			var neuronFactory = new RestrictedBoltzmannMachineFactory(RbmType.BinaryBinary, VisibleStatesCount, HiddenStatesCount, 
				DistributionType.Uniform, visibleUnitsProbability);
            _neuralNet = (RestrictedBoltzmannMachine) neuronFactory.CreateNeuralNet();
        }

	    private float[] CalculateVisibleUnitsProbability() {
		    var result = new float[_trainData[0].InputLength];
		    var factor = 1.0f/_trainData.Count;
		    foreach (var example in _trainData) {
			    var input = example.Input;
			    for (var i = 0; i < input.Length; i++) {
				    result[i] += ((Math.Abs(input[i] - TrueValue) < float.Epsilon) ? 1f : 0f)*factor;
			    }
		    }
		    return result;
	    }

		private void CompleteLearningOfNeuronNet() {
			var visibleUnitsProbability = CalculateVisibleUnitsProbability();
			
			CreateNeuronNet(visibleUnitsProbability);
			TrainNeuralNet(visibleUnitsProbability);
			TestNeuralNet();
			CalculateWeightsStat();
		}
        
        private void TrainNeuralNet(float[] visibleUnitsProbability) {
            const int iterationCount = 15;
            _trainProperties = new TrainProperties {
        		Epsilon = 0.001f,
        		MaxIterationCount = iterationCount,
        		Metrics = new HammingDistance(),
        		PackageSize = 20,
				CvLimit = 10.0f,
				BaseLearnSpeed = 0.001f,
				Momentum = 0.94f,
				LearnFactorStrategy = new LinearFactor(1f, 1f/30f, iterationCount),
                //LearnFactorStrategy = new ConstantFactor(),
				
				AddedLearnFactorStrategy = new LinearFactor(1f, 0.5f, iterationCount),

				SpeedBonus = 0.01f,
				SpeedPenalty = 0.99f,
				SpeedUpBorder = float.MaxValue,
				SpeedLowBorder = float.MinValue,
				AverageLearnFactor = 0.6f,

                //Regularization = new NoRegularization()
        		//Regularization = new EliminationRegularization(0.001f, 1.3f)
				Regularization = new L2Regularization(0.01f)
        	};

            var gradientFunction = new CenteredGradient(0.5f,
                                                        visibleUnitsProbability,
                                                        Enumerable.Repeat(0.5f, HiddenStatesCount).ToArray());
            //var gradientFunction = new LinearGradient();

	        //var trainMethod = new ContrastiveDivergence(_trainData, _testData, new LinearGradient(), 1);
			var trainMethod = new NativeWrapper.ContrastiveDivergenceNative(_trainData, _testData, gradientFunction, 1);
            trainMethod.InitilazeMethod(_neuralNet, _trainProperties);
            trainMethod.IterationCompleted += TrainingIterationCompleted;

            _stopWatch.Reset();
            _stopWatch.Start();
            trainMethod.Start();
            _stopWatch.Stop();
        }

        private void TrainingIterationCompleted (object sender, IterationCompletedEventArgs e) {
            var pointTrain = new Point(e.IterationNum, e.IterationValue);
            _sourceTrain.AppendAsync(Dispatcher, pointTrain);

	        if (!float.IsNaN(e.AddedIterationValue)) {
		        var pointTest = new Point(e.IterationNum, e.AddedIterationValue);
				_sourceTest.AppendAsync(Dispatcher, pointTest);
	        }
        }

        private void ListBoxTestPairsSelectionChanged(object sender, SelectionChangedEventArgs e) {
            var imageIndex = listBoxTestPairs.SelectedIndex;
	        ShowSourcePicture(_sourceTestImages[imageIndex]);
			
			_neuralNet.Predict(_testData[imageIndex].Input, _neuralNetOutput, false);
			ShowRecoveryPicture(_neuralNetOutput);
        }

		private void ShowSourcePicture (byte[] imageData) {
			const int stride = ImageSize;
            var bitmapImage = BitmapSource.Create(ImageSize, ImageSize, 96d, 96d, PixelFormats.Gray8, null, imageData, stride);

			imageSource.Source = bitmapImage;
        }

        private void ShowRecoveryPicture (float[] output) {
        	var imageData = ConvertToByteArray(output);
			const int stride = ImageSize;
            var bitmapImage = BitmapSource.Create(ImageSize, ImageSize, 96d, 96d, PixelFormats.Gray8, null, imageData, stride);
			imageRecovery.Source = bitmapImage;
        }

		private static byte[] ConvertToByteArray(float[] output) {
			var maxValue = output.Max();
			var minValue = output.Min();
			
			var array = new byte[output.Length];
			var j = 0;
			for (var i = 0; i < output.Length; i++) {
				array[j++] = (byte) (255.0 - ((output[i] - minValue)*255.0/(maxValue - minValue)));
			}
			return array;
		}

		private void TestNeuralNet() {
			Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => {
				StatusBarItemTime.Content = "Total time: " + _stopWatch.Elapsed.TotalSeconds + "s";
			}));
		}

		private void CalculateWeightsStat() {
			var weights = _neuralNet.Weights;
			var weightDoublePrecition = new double[weights.Length];
			var outputCurWeights = new StreamWriter(WeightsStatPath + "WeightsOfRBM.txt");
			for (var j = 0; j < weights.Length; j++) {
				outputCurWeights.WriteLine(weights[j]);
				weightDoublePrecition[j] = weights[j];
			}
			outputCurWeights.Close();

			var visibleBias = _neuralNet.VisibleStatesBias;
			var outputCurVisBias = new StreamWriter(WeightsStatPath + "VisibleBiasOfRBM.txt");
			for (var i = 0; i < visibleBias.Length; i++) {
				outputCurVisBias.WriteLine(visibleBias[i]);
			}
			outputCurVisBias.Close();

			var hiddenBias = _neuralNet.HiddenStatesBias;
			var outputCurHidBias = new StreamWriter(WeightsStatPath + "HiddenBiasOfRBM.txt");
			for (var i = 0; i < hiddenBias.Length; i++) {
				outputCurHidBias.WriteLine(hiddenBias[i]);
			}
			outputCurHidBias.Close();

			var statistics = new DescriptiveStatistics(weightDoublePrecition);
			Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => {
				StatusBarItemMean.Content = "Mean: " + statistics.Mean;
				StatusBarItemSd.Content = "Sd: " + statistics.StandardDeviation;
				StatusBarItemMin.Content = "Min: " + statistics.Minimum;
				StatusBarItemMax.Content = "Max: " + statistics.Maximum;
			}));

			const int vectorSize = ImageSize*ImageSize;
			var blocksCount = (int) Math.Sqrt(HiddenStatesCount);
			var pixels = new byte[weights.Length];

			for (var i = 0; i < blocksCount; i++) {
				for (var j = 0; j < blocksCount; j++) {
					var index = i*blocksCount + j;

					var localMax = float.MinValue;
					var localMin = float.MaxValue;
					for (var k = 0; k < vectorSize; k++) {
						var weight = weights[index*vectorSize + k];
						if (weight > localMax) {
							localMax = weight;
						}
						else if (weight < localMin) {
							localMin = weight;
						}
					}

					for (var m = 0; m < ImageSize; m++) {
						for (var n = 0; n < ImageSize; n++) {
							var weight = weights[index*vectorSize + m*ImageSize + n];
							pixels[i*blocksCount*vectorSize + j*ImageSize + m*blocksCount*ImageSize + n ] = (byte) ((weight - localMin)*255.0/(localMax - localMin));
						}
					}
				}
			}
			var bitmapImage = BitmapSource.Create(ImageSize*blocksCount, ImageSize*blocksCount, 96d, 96d, PixelFormats.Gray8, null, pixels, ImageSize*blocksCount);
			var stream = new FileStream(WeightsStatPath + "weights.png", FileMode.Create);
			var encoder = new PngBitmapEncoder();
			encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
			encoder.Save(stream);
			stream.Close();
		}

		private void WindowClosing(object sender,System.ComponentModel.CancelEventArgs e) {
			_workingThread.Abort();
		}
    }
}