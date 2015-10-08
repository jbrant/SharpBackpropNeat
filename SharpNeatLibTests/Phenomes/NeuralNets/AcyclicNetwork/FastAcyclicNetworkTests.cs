#region

using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNeat.Decoders;
using SharpNeat.Genomes.Neat;
using SharpNeatLibTests.Helper;

#endregion

namespace SharpNeat.Phenomes.NeuralNets.Tests
{
    [TestClass]
    public class FastAcyclicNetworkTests
    {
        private const string _genomeFile = "/Resources/5Out-3Hidden-5Out.gnm.xml";
        private string _inputFilePath;

        [TestInitialize]
        public void SetupTest()
        {
            _inputFilePath = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + _genomeFile;
        }

        [TestMethod]
        public void ActivateTest()
        {
            // Read in the NEAT genome
            NeatGenome genome = GenomeHelper.ReadNeatGenome(_inputFilePath, 5, 5);

            // Decode the genome to an acyclic network
            FastAcyclicNetwork network = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(genome);

            // Set the input array
            network.InputSignalArray[0] = 3;
            network.InputSignalArray[1] = 8;
            network.InputSignalArray[2] = -5;
            network.InputSignalArray[3] = 12;
            network.InputSignalArray[4] = 21;

            double curError = 0.0;
            for(int count = 0; count < 10000; count++)
            {
                // Activate the network
                network.Activate();

                curError = network.CalculateError(1);
            }
        }
    }
}