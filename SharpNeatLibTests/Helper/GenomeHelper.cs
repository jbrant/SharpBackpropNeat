#region

using System.Xml;
using SharpNeat.Genomes.Neat;
using SharpNeat.Network;

#endregion

namespace SharpNeatLibTests.Helper
{
    public static class GenomeHelper
    {
        public static NeatGenome ReadNeatGenome(string serializedGenomePath, int inputCount, int outputCount)
        {
            // Create a new genome factory
            NeatGenomeFactory genomeFactory = new NeatGenomeFactory(inputCount, outputCount,
                DefaultActivationFunctionLibrary.CreateLibraryNeat(PlainSigmoid.__DefaultInstance));

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
    }
}