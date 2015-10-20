#region

using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNeat.Decoders;
using SharpNeat.Genomes.Neat;
using SharpNeatLibTests.Helper;
using System.Collections.Generic;
using SharpNeat.Utility;
using System.Linq;

#endregion

namespace SharpNeat.Phenomes.NeuralNets.Tests
{
    [TestClass]
    public class HyperNEATAutoEncoderTests
    {
        private const string _genomeFileIn = "/Resources/CPPNS/NewToTest.gnm.xml";
        private const string _genomeFileOut = "/Resources/CPPNS/Out/champ_77,22_20151013_143931.gnm.xml";
        private string _inputFilePath;

        [TestInitialize]
        public void SetupTest()
        {
            _inputFilePath = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + _genomeFileIn;
        }

        [TestMethod]
        public void SaveImagesAsBPRuns()
        {
            SaveImagesAsBPRuns(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data",
                28, 100, 2, 100);
        }

        /// <summary>
        /// Create a network(from a CPPN) and runs back BP on it @loopNum times. It also saves the image created every 10th of the way through the problem
        /// ASSUME THE IMAGES TRAINED ON ARE SQUARES
        /// </summary>
        /// <param name="trainingImagesPath"> the location of the images to train on</param>
        /// <param name="visualFieldResolution">the size of one side of the square images provided in @trainingImagesPath</param>
        /// <param name="numImageSamples">Number of images samples to train on</param>
        /// <param name="reduceAmountPerSide">Reduce the resolution of eahc side by this value(MUST DIVIDE EVENLY, RECOMMENDED VALUE OF 2)</param>
        public void SaveImagesAsBPRuns(string trainingImagesPath, int visualFieldResolution, int numImageSamples, int reduceAmountPerSide, int loopNum)
        {
            // Read in the NEAT genome
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, 10, 2);
            IBlackBox phenome = GenomeHelper.CreateGenomeDecoder(visualFieldResolution / reduceAmountPerSide).Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            double maxFitness = numImageSamples * visualFieldResolution * visualFieldResolution;
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, visualFieldResolution * visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, visualFieldResolution);

            double oldError = 10000;
            for (int i = 0; i < loopNum; i++)
            {
                double[] trainingImageSample = allImageSamples[i % numImageSamples];
                phenome.ResetState(); ;

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                }
                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);

                phenome.Activate();
                if (i % (loopNum/10) == 0)
                {
                    ImageIoUtils.WriteImage(@"DataBPOut/Example" + (i++)/(loopNum/10) + ".bmp", phenome.OutputSignalArray);
                }
                double newError = phenome.CalculateError(4) * 100000;
                if (newError > oldError)
                {
                    System.Diagnostics.Debug.WriteLine("\n\n\n\n\n\n\n\newError > oldError: " + i);
                }

                if (i % (loopNum/ 10) == 0)
                {
                    // Activate the network
                    System.Diagnostics.Debug.WriteLine("\nError: " + newError);

                }
                oldError = newError;
            }

            #region Save Resulting Genome
            /*
            string pathForFullyConnected = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/FullConnected2Reduced_ForTest_DONTDELETE_DONTUSE.gnm.xml";
            NeatGenome genomeToSave = GenomeHelper.ReadStandardGenome(pathForFullyConnected, 14 * 14, 14 * 14);
            FastAcyclicNetwork networkForSave = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(genomeToSave);

            GenomeHelper.ChangeGenomeWeights(genomeToSave, ((FastAcyclicNetwork)phenome)._connectionArr);
            GenomeHelper.WriteNeatGenome(Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/SavedAfterBPGenomes/FullConnected2Reduced_SavedAfterNPTest.gnm.xml", genomeToSave);
            */
            #endregion
        }

        [TestMethod]
        public void SaveUpdatedVersionOfEachImage()
        {
           // SaveUpdatedVersionOfEachImage(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data",
           //     visualFieldResolution: 28, numImageSamples: 50, reduceAmountPerSide: 2, trainingSampleProportion: .8, numBackpropIterations: 100, learningRate: 1);

            RunBPUntilThresholdIsPassedThenSave(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data",
                visualFieldResolution: 28, numImageSamples: 50, reduceAmountPerSide: 2, trainingSampleProportion: .8, numBackpropIterations: 100, learningRate: 1, leavingThreshold: .99);
        }

        /// <summary>
        /// Run BP a fixed number of Times and then save the images that the Autoencoder creates for all images(training and validation)
        /// </summary>
        /// <param name="trainingImagesPath">The path for the images that are trained/used</param>
        /// <param name="visualFieldResolution">The visual field's pixel resolution, e.g. 11 means 11x11 pixels.</param>
        /// <param name="numImageSamples"></param>
        public void SaveUpdatedVersionOfEachImage(string trainingImagesPath, int visualFieldResolution, int numImageSamples, int reduceAmountPerSide, double trainingSampleProportion, int numBackpropIterations, double learningRate)
        {
            // Read in the NEAT genome
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, 6, 2);
            IBlackBox phenome = GenomeHelper.CreateGenomeDecoder(visualFieldResolution/ reduceAmountPerSide).Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            double maxFitness = numImageSamples * visualFieldResolution * visualFieldResolution;
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, visualFieldResolution * visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, visualFieldResolution);

            int trainingSampleEndIndex = (int)(allImageSamples.Count * trainingSampleProportion) - 1;
            List<double[]>  trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            List<double[]> validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();

            int i = 0;
            for (int iter = 0; iter < numBackpropIterations; iter++)
            {
                // Evaluate on each training sample
                foreach (double[] trainingImageSample in trainingImageSamples)
                {
                    // Reset the network
                    phenome.ResetState();

                    // Load the network inputs
                    for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                    {
                        phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                    }

                    // After inputs have been loaded, activate the network
                    phenome.Activate();

                    // Calculate the overall error based on how closely the outputs match the inputs
                    phenome.CalculateError(learningRate);
                }
            }

            foreach (double[] trainingImageSample in trainingImageSamples)
            {
                // Reset the network
                phenome.ResetState();

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                }
                ImageIoUtils.WriteImage(@"DataIn/Example" + i + ".bmp", trainingImageSample);
                    // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);
                ImageIoUtils.WriteImage(@"DataOut/Example" + i++ + "T.bmp", phenome.OutputSignalArray);
            }
            foreach (double[] validImageSample in validationImageSamples)
            {
                // Reset the network
                phenome.ResetState();

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < validImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = validImageSample[pixelIdx];
                }
                ImageIoUtils.WriteImage(@"DataIn/Example" + i + ".bmp", validImageSample);
                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);
                ImageIoUtils.WriteImage(@"DataOut/Example" + i++ + "V.bmp", phenome.OutputSignalArray);
            }
            System.Diagnostics.Debug.WriteLine((maxFitness - errorSum)/ maxFitness * 100);

            #region Save Resulting Genome
            /*
            string pathForFullyConnected = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/FullConnected2Reduced_ForTest_DONTDELETE_DONTUSE.gnm.xml";
            NeatGenome genomeToSave = GenomeHelper.ReadStandardGenome(pathForFullyConnected, 14 * 14, 14 * 14);
            FastAcyclicNetwork networkForSave = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(genomeToSave);

            GenomeHelper.ChangeGenomeWeights(genomeToSave, ((FastAcyclicNetwork)phenome)._connectionArr);
            GenomeHelper.WriteNeatGenome(Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/SavedAfterBPGenomes/FullConnected2Reduced_SavedAfterNPTest.gnm.xml", genomeToSave);
            */
            #endregion
        }

        /// <summary>
        /// Run BP until a performance threshold is met (or a limit of iterations is met) and then save the images that the Autoencoder creates for all images(training and validation)
        /// </summary>
        /// <param name="trainingImagesPath">The path for the images that are trained/used</param>
        /// <param name="visualFieldResolution">The visual field's pixel resolution, e.g. 11 means 11x11 pixels.</param>
        /// <param name="numImageSamples">Number of samples to read from trainingImagesPath</param>
        /// <param name="leavingThreshold">percent accuracy the autencoder must pass to prematurly stop BP</param>
        public void RunBPUntilThresholdIsPassedThenSave(string trainingImagesPath, int visualFieldResolution, int numImageSamples, int reduceAmountPerSide, double trainingSampleProportion, int numBackpropIterations, double learningRate, double leavingThreshold)
        {
            // Read in the NEAT genome
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, 10, 2);
            IBlackBox phenome = GenomeHelper.CreateGenomeDecoder(visualFieldResolution / reduceAmountPerSide).Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, visualFieldResolution * visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, visualFieldResolution);

            int trainingSampleEndIndex = (int)(allImageSamples.Count * trainingSampleProportion) - 1;
            List<double[]> trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            List<double[]> validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();
            double maxFitness = validationImageSamples.Count * visualFieldResolution * visualFieldResolution;


            for (int iter = 0; iter < numBackpropIterations; iter++)
            {
                // Evaluate on each training sample
                foreach (double[] trainingImageSample in trainingImageSamples)
                {
                    // Reset the network
                    phenome.ResetState();

                    // Load the network inputs
                    for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                    {
                        phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                    }

                    // After inputs have been loaded, activate the network
                    phenome.Activate();

                    // Calculate the overall error based on how closely the outputs match the inputs
                    phenome.CalculateError(learningRate);
                }

                errorSum = 0;
                // Now we're going to validate how well the network performs on the validation set
                foreach (double[] validationImageSample in validationImageSamples)
                {
                    // Reset the network
                    phenome.ResetState();

                    // Load the network inputs
                    for (int pixelIdx = 0; pixelIdx < validationImageSample.Length; pixelIdx++)
                    {
                        phenome.InputSignalArray[pixelIdx] = validationImageSample[pixelIdx];
                    }

                    // After inputs have been loaded, activate the network
                    phenome.Activate();

                    // Calculate the overall error *only once* based on how closely the outputs match the inputs                
                    errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                        phenome.OutputSignalArray);
                }
                double fitnessPerc = (maxFitness - errorSum) / maxFitness;
                System.Diagnostics.Debug.WriteLine("Iteration: " + iter + " With fitnessPerc: " + fitnessPerc);
                if (fitnessPerc > leavingThreshold)
                {
                    System.Diagnostics.Debug.WriteLine("Leaving at iteration: " + iter +" With threshold: " + leavingThreshold);
                    break;
                }
            }

            int i = 0;
            foreach (double[] trainingImageSample in trainingImageSamples)
            {
                // Reset the network
                phenome.ResetState();

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                }
                ImageIoUtils.WriteImage(@"DataIn/Example" + i + ".bmp", trainingImageSample);
                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);
                ImageIoUtils.WriteImage(@"DataOut/Example" + i++ + "T.bmp", phenome.OutputSignalArray);
            }

            foreach (double[] validImageSample in validationImageSamples)
            {
                // Reset the network
                phenome.ResetState();

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < validImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = validImageSample[pixelIdx];
                }
                ImageIoUtils.WriteImage(@"DataIn/Example" + i + ".bmp", validImageSample);
                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);
                ImageIoUtils.WriteImage(@"DataOut/Example" + i++ + "V.bmp", phenome.OutputSignalArray);
            }
            System.Diagnostics.Debug.WriteLine((maxFitness - errorSum) / maxFitness * 100);
            #region Save Resulting Genome

            string pathForFullyConnected = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/FullConnected2Reduced_ForTest_DONTDELETE_DONTUSE.gnm.xml";
            NeatGenome genomeToSave = GenomeHelper.ReadStandardGenome(pathForFullyConnected, 14 * 14, 14 * 14);
            FastAcyclicNetwork networkForSave = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(genomeToSave);

            GenomeHelper.ChangeGenomeWeights(genomeToSave, ((FastAcyclicNetwork)phenome)._connectionArr);
            GenomeHelper.WriteNeatGenome(Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"/Resources/SavedAfterBPGenomes/FullConnected2Reduced_SavedAfterNPTest.gnm.xml", genomeToSave);
            #endregion
        }

        /// <summary>
        /// Reads in images and saves those images
        /// If there's an error, you need to create the associated folder
        /// </summary>
        [TestMethod]
        public void PureInputAndOutputForComparison()
        {
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data", 28 * 28, 20, 255);

            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, 2, 28);
            int i = 0;
            foreach (double[] trainingImageSample in allImageSamples)
            {
                ImageIoUtils.WriteImage(@"DataIn/Example" + i++ + ".bmp", trainingImageSample);
            }
        }
    }
}