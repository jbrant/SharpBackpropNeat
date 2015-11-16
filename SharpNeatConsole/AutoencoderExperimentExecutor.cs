/* ***************************************************************************
 * This file is part of SharpNEAT - Evolution of Neural Networks.
 * 
 * Copyright 2004-2006, 2009-2010 Colin Green (sharpneat@gmail.com)
 *
 * SharpNEAT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SharpNEAT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SharpNEAT.  If not, see <http://www.gnu.org/licenses/>.
 */

#region

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml;
using log4net.Config;
using SharpNeat;
using SharpNeat.Core;
using SharpNeat.Domains.EvolvedAutoEncoderHyperNeat;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Utility;

#endregion

namespace SharpNeatConsole
{
    /// <summary>
    ///     Minimal console application that hardwaires the setting up on a evolution algorithm and start it running.
    /// </summary>
    internal class AutoencoderExperimentExecutor
    {
        private static void Main(string[] args)
        {
            if (args == null || args.Length != 4)
            {
                Console.Error.WriteLine(
                    "Invocation of the form: sharpneatconsole {experiment configuration file} {# runs} {output directory} {experiment base name}");
                throw new SharpNeatException("Malformed invocation.");
            }

            // Read in experiment configuration file
            string experimentConfigurationFile = args[0];
            int numRuns = Int32.Parse(args[1]);
            _baseOutputDirectory = args[2];
            _experimentBaseName = args[3];

            // Initialise log4net (log to console).
            XmlConfigurator.Configure(new FileInfo("log4net.properties"));

            // Create output directory
            Directory.CreateDirectory(Path.Combine(_baseOutputDirectory, _experimentBaseName));

            // Configure XML writer
            _xwSettings = new XmlWriterSettings();
            _xwSettings.Indent = true;

            // Initialize experiment
            _autoencoderExperiment = new EvolvedAutoEncoderHyperNeatExperiment();

            // Load experiment configuration
            XmlDocument experimentConfig = new XmlDocument();
            experimentConfig.Load(experimentConfigurationFile);
            _autoencoderExperiment.Initialize("Evolved Autoencoder", experimentConfig.DocumentElement);

            Console.WriteLine(@"Executing Experiment with parameters defined in {0}", experimentConfigurationFile);

            for (_curRun = 0; _curRun < numRuns; _curRun++)
            {
                string runBaseLogFileName = _experimentBaseName + "_Run" + (_curRun + 1) + '_';

                // Confiure log file writer
                string logFilename = runBaseLogFileName + '_' + DateTime.Now.ToString("yyyyMMdd") + ".csv";
                _logFileWriter =
                    new StreamWriter(
                        Path.Combine(_baseOutputDirectory, _experimentBaseName, logFilename), true);
                _logFileWriter.WriteLine(
                    "ClockTime,Gen,BestFitness,MeanFitness,MeanSpecieChampFitness,ChampComplexity,MeanComplexity,MaxComplexity,TotalEvaluationCount,EvaluationsPerSec,SearchMode");

                // Create a new genome factory
                _genomeFactory = _autoencoderExperiment.CreateGenomeFactory();

                // Create an initial population of 150 randomly generated genomes.
                _genomeList = _genomeFactory.CreateGenomeList(150, 0);

                // Create evolution algorithm and attach update event.
                _ea = _autoencoderExperiment.CreateEvolutionAlgorithm(_genomeFactory, _genomeList);
                _ea.UpdateEvent += ea_UpdateEvent;

                // Set the current generation
                _curGenerationUpdate = 0;

                Console.WriteLine(@"Executing Run {0} of {1}", _curRun + 1, numRuns);
                
                // Start algorithm (it will run on a background thread).
                _ea.StartContinue();

                while (RunState.Terminated != _ea.RunState && RunState.Paused != _ea.RunState)
                {
                    Thread.Sleep(200);
                }
                
                // Build image output directory name
                string imageOutputDirectory = string.Format("Run{0}_{1}", _curRun + 1, _imageOutputDirectoryBaseName);

                // Create image output directory
                Directory.CreateDirectory(Path.Combine(_baseOutputDirectory, _experimentBaseName, imageOutputDirectory));
                
                // Build path to last genome file
                string lastGenomePath = Path.Combine(_baseOutputDirectory, _experimentBaseName,
                    string.Format("ChampGenome_{0}_Run{1}_Generation{2}.gnm.xml",
                        _experimentBaseName,
                        _curRun + 1, _curGenerationUpdate));
                        
                // Backpropagate over last genome and save image results
                RunBpAndSave(lastGenomePath, imageOutputDirectory, _autoencoderExperiment.InputCount,
                    _autoencoderExperiment.OutputCount,
                    _autoencoderExperiment._trainingImagesFilename, _autoencoderExperiment.VisualFieldResolution,
                    _imageSampleCount, _resolutionReduction,
                    _trainingSampleProportion, _numBackpropIterations, _learningRate);
            }
        }

        private static void ea_UpdateEvent(object sender, EventArgs e)
        {
            Console.WriteLine("gen={0:N0} bestFitness={1:N6}", _ea.CurrentGeneration, _ea.Statistics._maxFitness);

            if (_ea.CurrentGeneration > _curGenerationUpdate)
            {
                // Derive best genome filename
                string bestGenomeFile = string.Format("ChampGenome_{0}_Run{1}_Generation{2}.gnm.xml",
                    _experimentBaseName,
                    _curRun + 1, _ea.CurrentGeneration);

                // Save genome to xml file.
                using (
                    XmlWriter xw =
                        XmlWriter.Create(
                            Path.Combine(_baseOutputDirectory, _experimentBaseName, bestGenomeFile),
                            _xwSettings))
                {
                    _autoencoderExperiment.SavePopulation(xw, new[] {_ea.CurrentChampGenome});
                }

                // Write statistics
                _logFileWriter.WriteLine("{0:yyyy-MM-dd HH:mm:ss.fff},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}",
                    DateTime.Now, _ea.Statistics._generation, _ea.Statistics._maxFitness, _ea.Statistics._meanFitness,
                    _ea.Statistics._meanSpecieChampFitness, _ea.CurrentChampGenome.Complexity,
                    _ea.Statistics._meanComplexity, _ea.Statistics._maxComplexity, _ea.Statistics._totalEvaluationCount,
                    _ea.Statistics._evaluationsPerSec, _ea.ComplexityRegulationMode);
                _logFileWriter.Flush();

                // Update current update generation
                _curGenerationUpdate = _ea.CurrentGeneration;
            }
        }

        private static void RunBpAndSave(string inputGenomePath, string imageOutputDirectoryName, int inputNeuronCount,
            int outputNeuronCount,
            string trainingImagesPath, int visualFieldResolution, int numImageSamples, int reduceAmountPerSide,
            double trainingSampleProportion, int numBackpropIterations, double learningRate)
        {
            // Read in the NEAT genome
            NeatGenome cppnGenome = GenomeHelper.ReadCPPNGenome(inputGenomePath, inputNeuronCount, outputNeuronCount);
            IBlackBox phenome = GenomeHelper.CreateGenomeDecoder(visualFieldResolution / reduceAmountPerSide, true).Decode(cppnGenome);
            // Decode the genome to an acyclic network

            double errorSum = 0;
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, visualFieldResolution * visualFieldResolution, numImageSamples, 255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, visualFieldResolution);

            int trainingSampleEndIndex = (int)(allImageSamples.Count * trainingSampleProportion) - 1;
            List<double[]> trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            List<double[]> validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();
            double maxFitness = validationImageSamples.Count * visualFieldResolution * visualFieldResolution / (reduceAmountPerSide * reduceAmountPerSide);


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

                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);

                ImageIoUtils.WriteImage(
                    Path.Combine(_baseOutputDirectory, _experimentBaseName,
                        imageOutputDirectoryName,
                        "Example" + i++ + "T.bmp"),
                    phenome.OutputSignalArray);
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

                // After inputs have been loaded, activate the network
                phenome.Activate();
                errorSum += BackpropagationUtils.CalculateOutputError(phenome.InputSignalArray,
                phenome.OutputSignalArray);

                ImageIoUtils.WriteImage(
                    Path.Combine(_baseOutputDirectory, _experimentBaseName,
                        imageOutputDirectoryName,
                        "Example" + i++ + "V.bmp"),
                    phenome.OutputSignalArray);
            }
            Console.WriteLine(@"Final Fitness: " + (maxFitness - errorSum) / maxFitness * 100);            
        }

        #region Internal static variables

        private static IGenomeFactory<NeatGenome> _genomeFactory;
        private static List<NeatGenome> _genomeList;
        private static INeatEvolutionAlgorithm<NeatGenome> _ea;
        private static NumberFormatInfo _filenameNumberFormatter;
        private static string _baseOutputDirectory;
        private static string _experimentBaseName;
        private static EvolvedAutoEncoderHyperNeatExperiment _autoencoderExperiment;
        private static XmlWriterSettings _xwSettings;
        private static StreamWriter _logFileWriter;
        private static uint _curGenerationUpdate;
        private static int _curRun;

        #endregion

        #region Hard-coded parameters for image generation

        private static readonly int _imageSampleCount = 150;
        private static readonly int _resolutionReduction = 1;
        private static readonly double _trainingSampleProportion = 0.8;
        private static readonly int _numBackpropIterations = 100;
        private static readonly int _learningRate = 1;
        private static readonly string _imageOutputDirectoryBaseName = "ImageOutput";

        #endregion
    }
}