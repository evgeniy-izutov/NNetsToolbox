using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNet.MultyLayerPerceptron {
    public sealed class MultyLayerPerceptron : INeuralNet {
        private BaseNeuralBlock[] _layers;
        private const int FirstLayerNum = 0;
        private const int SecondLayerNum = 1;
        private int _lastLayerNum;

        public MultyLayerPerceptron() {}

        public MultyLayerPerceptron(int layersCount) {
            _lastLayerNum = layersCount - 1;
            _layers = new BaseNeuralBlock[layersCount];
        }

        public void AddNeuralBlock(BaseNeuralBlock block, int layerNum) {
            if ((layerNum < 0) && (layerNum >= _layers.Length)) {
                throw new ArgumentOutOfRangeException("No memory for new block in layer");
            }
            _layers[layerNum] = block;
        }

        public void Predict(float[] input, float[] output) {
            CalculateFirstLayer(input);
            CalculateLeftoverLayers();
            SetOutput(output);
        }

        public void Save(string outputPath) {
            var outputStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
            var serializer = new BinaryFormatter();
            serializer.Serialize(outputStream, _layers);
            outputStream.Close();
        }

        public void Load(string inputPath) {
            var inputStream = new FileStream(inputPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            var deserializer = new BinaryFormatter();
            _layers = (BaseNeuralBlock[]) deserializer.Deserialize(inputStream);
            _lastLayerNum = _layers.Length - 1;
            inputStream.Close();
        }

        public BaseNeuralBlock[] Layers {
            get { return _layers; }
        }

        public int[] GetLayersStruct() {
            var layersStruct = new int[_layers.Length + 1];
            layersStruct[0] = _layers[0].GetWeights()[0].Length/_layers[0].Size;
            for (var i = 0; i < _layers.Length; i++) {
                layersStruct[i + 1] = _layers[i].Size;
            }
            return layersStruct;
        }

        private void CalculateFirstLayer(float[] input) {
            _layers[FirstLayerNum].Calculate(input);
        }

        private void CalculateLeftoverLayers() {
            for (var layerNum = SecondLayerNum;
                 layerNum < _layers.Length;
                 layerNum++) {
                _layers[layerNum].Calculate();
            }
        }

        private void SetOutput(float[] output) {
            var neuronNetOutput = _layers[_lastLayerNum].GetState();
            for (var i = 0; i < output.Length; i++) {
                output[i] = neuronNetOutput[i];
            }
        }
    }
}