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
using SharpNeat.Decoders.HyperNeat;
using SharpNeat.Core;
using SharpNeat.Network;
using System.Xml;

#endregion

namespace SharpNeat.Phenomes.NeuralNets.Tests
{
    [TestClass]
    public class HyperNEATAutoEncoderTests
    {
        private const string _genomeFileIn = "/Resources/CPPNS/NewToTest.gnm.xml";
        private const string _genomeFileOut = "/Resources/CPPNS/Out/champ_77,22_20151013_143931.gnm.xml";
        private const string _substrateFileOut = "/Resources/CPPNS/Out/substrate.gnm.xml";
        private string _inputFilePath;
        private Substrate _substrate;
        private HyperNeatDecoder _genomeDecoder;
        private int _visualFieldResolution = 28;
        private int _reduceAmountPerSide = 4;
        private int _inputNeuronCount = 7;
        private int _outpurNeuronCount = 6;
        private double _learningRate = 1;
        private string _trainingImagesPath = @"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data";

        [TestInitialize]
        public void SetupTest()
        {
            _inputFilePath = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + _genomeFileIn;
            _substrate = GenomeHelper.CreateSubstrate(_visualFieldResolution / _reduceAmountPerSide, true);
            _genomeDecoder = (HyperNeatDecoder) GenomeHelper.CreateGenomeDecoder(_substrate, true);
        }

        [TestMethod]
        public void SaveImagesAsBPRuns()
        {
            SaveImagesAsBPRuns(100, 100);
        }
        
        /// <summary>
        /// Create a network(from a CPPN) and runs back BP on it @loopNum times. It also saves the image created every 10th of the way through the problem
        /// ASSUME THE IMAGES TRAINED ON ARE SQUARES
        /// </summary>
        /// <param name="numImageSamples">Number of images samples to train on</param>
        public void SaveImagesAsBPRuns(int numImageSamples, int loopNum)
        {
            // Read in the NEAT genome
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, _inputNeuronCount, _outpurNeuronCount);
            IBlackBox phenome = _genomeDecoder.Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            double maxFitness = 1 * _visualFieldResolution * _visualFieldResolution/(_reduceAmountPerSide *_reduceAmountPerSide);
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(_trainingImagesPath, _visualFieldResolution * _visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, _reduceAmountPerSide, _visualFieldResolution);

            double oldError = 10000;
            for (int i = 0; i < loopNum; i++)
            {
                double[] trainingImageSample = allImageSamples[1];
                phenome.ResetState(); ;

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                }

                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum = BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);

                //System.Diagnostics.Debug.WriteLine((maxFitness - errorSum) / maxFitness * 100);
                if (loopNum / 10 == 0 || i % (loopNum/10) == 0)
                {
                    ImageIoUtils.WriteImage(@"DataBPOut/Example" + (i) + ".bmp", phenome.OutputSignalArray);
                    System.Diagnostics.Debug.WriteLine("\n" + i + "Error: " + errorSum);
                }
                double newError = phenome.CalculateError(_learningRate) * 100000;
                if (newError > oldError)
                {
                    System.Diagnostics.Debug.WriteLine("\n\nnewError > oldError: " + i);
                    //Assert.IsTrue(newError < oldError);
                }

                oldError = newError;
            }
        }

        [TestMethod]
        public void SaveUpdatedVersionOfEachImage()
        {
           // SaveUpdatedVersionOfEachImage(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data",
           //     visualFieldResolution: 28, numImageSamples: 50, reduceAmountPerSide: 2, trainingSampleProportion: .8, numBackpropIterations: 100, learningRate: 1);

            RunBPUntilThresholdIsPassedThenSave(@"C:\Users\Christopher\Documents\GitHub\SharpBackpropNeat\SharpNeatDomains\EvolvedAutoencoder\ImageData\Number1Samples.data",
                visualFieldResolution: 28, numImageSamples: 100, reduceAmountPerSide: 4, trainingSampleProportion: .8, numBackpropIterations: 100, learningRate: 1, leavingThreshold: 10);
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
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, _inputNeuronCount, _outpurNeuronCount);
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
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, _inputNeuronCount, _outpurNeuronCount);
            IBlackBox phenome = GenomeHelper.CreateGenomeDecoder(visualFieldResolution / reduceAmountPerSide, true).Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, visualFieldResolution * visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, visualFieldResolution);

            int trainingSampleEndIndex = (int)(allImageSamples.Count * trainingSampleProportion) - 1;
            List<double[]> trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            List<double[]> validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();
            double maxFitness = validationImageSamples.Count * visualFieldResolution * visualFieldResolution/(reduceAmountPerSide * reduceAmountPerSide);


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
                #region Save an image to show progress thorugh BP

                double[] trainingImageSampleForImage = allImageSamples[1];
                phenome.ResetState();

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSampleForImage.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSampleForImage[pixelIdx];
                }

                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum = BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);

                ImageIoUtils.WriteImage(@"DataBPOut/Example" + (iter) + "_ErrorOnImage" + errorSum + ".bmp", phenome.OutputSignalArray);
                //System.Diagnostics.Debug.WriteLine("\n" + iter + "Error: " + errorSum);// * /
                #endregion

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
                    //break;
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

            errorSum = 0;
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
        }

        #region Base Functionality
        /// <summary>
        /// Reads in images and saves those images
        /// If there's an error, you need to create the associated folder
        /// </summary>
        [TestMethod]
        public void PureInputAndOutputForComparison()
        {
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(_trainingImagesPath, _visualFieldResolution * _visualFieldResolution, 20, 255);

            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, _reduceAmountPerSide, _visualFieldResolution);
            int i = 0;
            foreach (double[] trainingImageSample in allImageSamples)
            {
                ImageIoUtils.WriteImage(@"DataIn/Example" + i++ + ".bmp", trainingImageSample);
            }
        }

        /// <summary>
        /// Save a network definition created from a CPPN
        /// </summary>
        [TestMethod]
        public void SaveSubstrateFromGenome()
        {
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(_inputFilePath, _inputNeuronCount, _outpurNeuronCount);
            IBlackBox phenome = _genomeDecoder.GeCPPNGenomeDecoder()(cppnGenome);

            string pathForFullyConnected = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + _substrateFileOut;
            XmlWriter writer = XmlWriter.Create(pathForFullyConnected);
            INetworkDefinition def = _substrate.CreateNetworkDefinition(phenome, true);
            NetworkXmlIO.WriteComplete(writer, def, true);
            writer.Close();
        }
        #endregion
    }
}