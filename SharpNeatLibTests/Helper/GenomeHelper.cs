#region

using System.Xml;
using SharpNeat.Genomes.Neat;
using SharpNeat.Network;
using SharpNeat.Phenomes.NeuralNets;
using SharpNeat.Core;
using SharpNeat.Phenomes;
using SharpNeat.Decoders.HyperNeat;
using System.Collections.Generic;
using SharpNeat.Decoders;

#endregion
//ToDo:ChangeGenomeWeights probably doens't need to return anything
namespace SharpNeatLibTests.Helper
{
    public static class GenomeHelper
    {
        #region Read Genome and associated wrappers
        public static NeatGenome ReadNeatGenome(string serializedGenomePath, int inputCount, int outputCount, IActivationFunctionLibrary actFuncLib)
        {
            // Create a new genome factory
            NeatGenomeFactory genomeFactory = new NeatGenomeFactory(inputCount, outputCount, actFuncLib);

            // Create a reader for the serialized genome
            XmlReader reader = XmlReader.Create(serializedGenomePath);

            // Create XML document and give it the reader reference
            XmlDocument document = new XmlDocument();
            document.Load(reader);

            // Traverse down to the network definition
            XmlNodeList nodeList = document.GetElementsByTagName("Root");

            // Read in the genome
            NeatGenome genome = NeatGenomeXmlIO.LoadCompleteGenomeList(nodeList[0], false, genomeFactory)[0];

            return genome;
        }

        public static NeatGenome ReadCPPNGenome(string serializedGenomePath, int inputCount, int outputCount)
        {
            return ReadNeatGenome(serializedGenomePath, inputCount, outputCount, DefaultActivationFunctionLibrary.CreateLibraryCppn());
        }

        public static NeatGenome ReadStandardGenome(string serializedGenomePath, int inputCount, int outputCount)
        {
            return ReadNeatGenome(serializedGenomePath, inputCount, outputCount, DefaultActivationFunctionLibrary.CreateLibraryNeat(PlainSigmoid.__DefaultInstance));
        }
        #endregion

        /// <summary>
        /// For saving and vizualization, we somtimes need to change the genome
        /// BackProp for instance, happens on the phenotype and so if we want to see/save those changes
        /// we need to send the data back to NeatGenome
        /// </summary>
        /// <param name="genomeToWrite">The genome whose weights are being changed</param>
        /// <param name="connections">we set the genome's weights to the weights of these connections</param>
        /// <returns></returns>
        public static NeatGenome ChangeGenomeWeights(NeatGenome genomeToWrite, FastConnection[] connections)
        {
            IConnectionList connectionList = genomeToWrite.ConnectionList;
            int connectionCount = connectionList.Count;

            for (int i = 0; i < connectionCount; i++)
            {
                genomeToWrite.ConnectionGeneList[i].Weight = connections[i]._weight;
            }
            return genomeToWrite;
        }

        public static void WriteNeatGenome(string serializedGenomePath, NeatGenome genomeToWrite)
        {
            // Create a reader for the serialized genome
            XmlWriter writer = XmlWriter.Create(serializedGenomePath);

            NeatGenomeXmlIO.WriteComplete(writer, genomeToWrite, true);
            writer.Close();
        }

        //Deal with later. Duplicate

        /// <summary>
        /// Creates a genome decoder. We split this code into a separate  method so that it can be re-used by the problem domain visualization code
        /// (it needs to decode genomes to phenomes in order to create a visualization).
        /// </summary>
        /// <param name="visualFieldResolution">The visual field's pixel resolution, e.g. 11 means 11x11 pixels.</param>
        /// <param name="lengthCppnInput">Indicates if the CPPNs being decoded have an extra input for specifying connection length.</param>
        public static IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder(int visualFieldResolution, bool lengthCppnInput = false)
        {
            // Create two layer 'sandwich' substrate.
            int pixelCount = visualFieldResolution * visualFieldResolution;
            double pixelSize = 2 / visualFieldResolution;
            double originPixelXY = -1 + (pixelSize / 2.0);

            SubstrateNodeSet inputLayer = new SubstrateNodeSet(pixelCount);
            SubstrateNodeSet outputLayer = new SubstrateNodeSet(pixelCount);

            // Node IDs start at 1. (bias node is always zero).
            uint inputId = 1;
            uint outputId = (uint)(pixelCount + 1);
            double yReal = originPixelXY;

            for (int y = 0; y < visualFieldResolution; y++, yReal += pixelSize)
            {
                double xReal = originPixelXY;
                for (int x = 0; x < visualFieldResolution; x++, xReal += pixelSize, inputId++, outputId++)
                {
                    inputLayer.NodeList.Add(new SubstrateNode(inputId, new double[] { xReal, yReal, -1.0 }));
                    outputLayer.NodeList.Add(new SubstrateNode(outputId, new double[] { xReal, yReal, 1.0 }));
                }
            }

            List<SubstrateNodeSet> nodeSetList = new List<SubstrateNodeSet>(2);
            nodeSetList.Add(inputLayer);
            nodeSetList.Add(outputLayer);

            // Define connection mappings between layers/sets.
            List<NodeSetMapping> nodeSetMappingList = new List<NodeSetMapping>(1);
            nodeSetMappingList.Add(NodeSetMapping.Create(0, 1, (double?)null));

            // Construct substrate.
            Substrate substrate = new Substrate(nodeSetList, DefaultActivationFunctionLibrary.CreateLibraryNeat(SteepenedSigmoid.__DefaultInstance), 0, 0.2, 5, nodeSetMappingList);

            // Create genome decoder. Decodes to a neural network packaged with an activation scheme that defines a fixed number of activations per evaluation.
            IGenomeDecoder<NeatGenome, IBlackBox> genomeDecoder = new HyperNeatDecoder(substrate, NetworkActivationScheme.CreateCyclicFixedTimestepsScheme(4), NetworkActivationScheme.CreateAcyclicScheme(), lengthCppnInput);
            return genomeDecoder;
        }

    }
}