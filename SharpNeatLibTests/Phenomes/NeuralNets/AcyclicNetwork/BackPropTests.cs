#region

using System;
using System.IO;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNeat.Decoders;
using SharpNeat.Genomes.Neat;
using SharpNeatLibTests.Helper;

#endregion

namespace SharpNeat.Phenomes.NeuralNets.Tests
{
    [TestClass]
    public class BackPropTests
    {
        private const string _genomeFile = "/Resources/5Out5In.gnm.xml";
        private string _inputFilePath;
        private double _learningRate = 1;

        [TestInitialize]
        public void SetupTest()
        {
            _inputFilePath = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + _genomeFile;
        }

        [TestMethod]
        public void ActivateTest()
        {
            // Read in the NEAT genome
            NeatGenome genome = GenomeHelper.ReadStandardGenome(_inputFilePath, 5, 5);

            // Decode the genome to an acyclic network
            FastAcyclicNetwork network = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(genome);

            // Set the input array
            network.InputSignalArray[0] = 0;
            network.InputSignalArray[1] = .2;
            network.InputSignalArray[2] = .3;
            network.InputSignalArray[3] = .6;
            network.InputSignalArray[4] = 1;

            double curError = 10000.0;
            for(int count = 0; count < 100; count++)
            {
                // Activate the network
                network.Activate();

                double newError = network.CalculateError(_learningRate);
                System.Diagnostics.Debug.WriteLine(newError);
                //Assert.IsTrue(newError < curError);
                curError = newError;

            }
            System.Diagnostics.Debug.WriteLine("\n\n");
            for (int i = 0; i < network.OutputCount; i++)
            {
                System.Diagnostics.Debug.WriteLine("" + network.OutputSignalArray[i]);
            }
        }
    }
}