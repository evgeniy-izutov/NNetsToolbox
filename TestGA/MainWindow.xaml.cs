using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Media;
using GeneticAlgorithm;
using Microsoft.Research.DynamicDataDisplay;
using Microsoft.Research.DynamicDataDisplay.DataSources;
using StandardTypes;

namespace TestGA {
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        private const float MinValue = -500.0f;
        private const float MaxValue = 500.0f;
        private const int MaxIterCount = 2000;
        private const int PopulationSzie = 1000;
        private const int GenotypeLength = 500;
        private const int TournamentSize = 5;
        private const float Alpha = 0.5f;
        private const float NewPopulationFactor = 0.6f;
        private const float ElitePopulationFactor = 0.1f;
        private const float CrossingProbability = 0.9f;
        private const float MutationProbobility = 0.1f;
		private readonly ObservableDataSource<Point> _learningGraph;

        public MainWindow() {
            InitializeComponent();
            
			_learningGraph = new ObservableDataSource<Point>();
            chartPlotter.Children.Add(new LineGraph(_learningGraph) {
                Stroke = Brushes.Gold,
                StrokeThickness = 2
            });
			
			var geneticAlgorithm = CreateGeneticAlgorithm();

            var stopWatch = new Stopwatch();
            stopWatch.Reset();

            stopWatch.Start();
            geneticAlgorithm.Start();
        	var result = geneticAlgorithm.GetResult();
            stopWatch.Stop();

            StatusBarItemTime.Content = "Time: " + stopWatch.Elapsed.TotalSeconds;
            ShowResultTable(result[0]);
        }

        private GeneticAlgorithm.GeneticAlgorithm CreateGeneticAlgorithm() {
            var random = new Random();
			var structure = new[] {GenotypeLength};
			var minValues = new[] {MinValue};
			var maxValues = new[] {MaxValue};

        	var initilizeProperties = new InitilizeProperties {
				PopulationsCount = 1,
				PopulationSize = PopulationSzie,
				NewPopulationFactor = NewPopulationFactor,
				ElitePopulationFactor = ElitePopulationFactor,
				IsEliteIndividualAlways = true,
				CrossingProbability = CrossingProbability,
				MutationProbobility = MutationProbobility,
				IterationCount = MaxIterCount,
				ChromosomesStructure = structure,
				FitnessFunction = new TestFitnessFunction(),
				SelectionOperator = new TournamentSelection(TournamentSize, random.Next()),
				CrossoverOperator = new BlXalphaCrossoverWithBorder(Alpha, minValues, maxValues, random.Next()),
				MutationOperator = new SingleMutation(minValues, maxValues, random.Next()),
				ChromosomesDistribution = new ChromosomesDistribution(DistributionType.Uniform, minValues, maxValues, random.Next()),
				Criterion = OptimizationCriterion.MinCriterion
			};
			var geneticAlgorithm = new GeneticAlgorithm.GeneticAlgorithm();
			geneticAlgorithm.InitilazeAlgorithm(initilizeProperties, random.Next());
        	geneticAlgorithm.IterationCompleted += CompleteTrainEpochHandler;
            return geneticAlgorithm;
        }

        private void ShowResultTable (float[] result) {
            var coll = new ObservableCollection<TableData>();

            for (var i = 0; i < result.Length; i++) {
                coll.Add(new TableData {
                        Num = i + 1,
                        Value = result[i],
                    });
            }

            dataGridResult.ItemsSource = coll;
            dataGridResult.Items.Refresh();
        }

		private void CompleteTrainEpochHandler (object sender, IterationCompletedEventArgs e) {
            var pointLearning = new Point(e.IterationNum + 1, e.IterationValue);
            _learningGraph.AppendAsync(Dispatcher, pointLearning);
        }
    }
}
