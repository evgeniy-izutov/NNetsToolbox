using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace StandardTypes {
	public sealed class MnistDataLoader : IDataLoader {
		private readonly string _dataFilePath;
		private readonly string _labelFilePath;
		private readonly bool _isBlackOnly;
		private readonly float _trueValue;
		private readonly float _falsevalue;
		private readonly Random _uniformGenerator;
		
		public MnistDataLoader(string dataFilePath, string labelFilePath, bool isBlackOnly, float trueValue, float falsevalue) {
			_dataFilePath = dataFilePath;
			_labelFilePath = labelFilePath;
			_isBlackOnly = isBlackOnly;
			_trueValue = trueValue;
			_falsevalue = falsevalue;
			_uniformGenerator = new Random();
		}
		
		public List<TrainSingle> LoadData(List<FeatureDescription> inputDescription) {
			var data = new List<TrainSingle>();
			
			using (var dataStream = new BinaryReader(File.Open(_dataFilePath, FileMode.Open))) {
				dataStream.ReadUInt32();
				
				var setSize = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var imageHaight = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var imageWidth = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var pixelsCount = imageHaight*imageWidth;

				var imageBuffer = new byte[pixelsCount];
				for (var i = 0; i < setSize; i++) {
					dataStream.Read(imageBuffer, 0, pixelsCount);

					var input = ConvertImageToArray(imageBuffer);
					var example = new TrainSingle(input);
					example.Id = i;
					example.Weight = 1.0f;

					data.Add(example);
				}
			}

			return data;
		}

		public List<TrainPair> LoadData(List<FeatureDescription> inputDescription, List<FeatureDescription> outputDescription) {
			var data = new List<TrainPair>();

			using (BinaryReader dataStream = new BinaryReader(File.Open(_dataFilePath, FileMode.Open)) , 
				   labelStream = new BinaryReader(File.Open(_labelFilePath, FileMode.Open))) {

				dataStream.ReadUInt32();
				labelStream.ReadUInt32();
				labelStream.ReadUInt32();
				
				var setSize = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var imageHaight = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var imageWidth = BitConverter.ToInt32(dataStream.ReadBytes(4).Reverse().ToArray(), 0);
				var pixelsCount = imageHaight*imageWidth;

				var imageBuffer = new byte[pixelsCount];
				for (var i = 0; i < setSize; i++) {
					dataStream.Read(imageBuffer, 0, pixelsCount);

					var label = labelStream.ReadByte();

					var input = ConvertImageToArray(imageBuffer);
					var output = ConvertLabelToArray(label);
					var example = new TrainPair(input, output);
					example.Id = i;
					example.Weight = 1.0f;

					data.Add(example);
				}
			}

			return data;
		}

		private float[] ConvertImageToArray(byte[] pixels) {
			const double factor = 1.0/255.0;
			var result = new float[pixels.Length];

			if (_isBlackOnly) {
				for (var i = 0; i < pixels.Length; i++) {
					result[i] = (_uniformGenerator.NextDouble() <= factor*pixels[i]) ? _trueValue : _falsevalue;
				}
			}
			else {
				for (var i = 0; i < pixels.Length; i++) {
					result[i] = pixels[i];
				}
			}
			
			return result;
		}

		private float[] ConvertLabelToArray(byte label) {
			const int size = 10;
			var result = new float[size];
			for (var i = 0; i < label; i++) {
				result[i] = _falsevalue;
			}
			result[label] = _trueValue;
			for (var i = label + 1; i < size; i++) {
				result[i] = _falsevalue;
			}
			return result;
		}
	}
}