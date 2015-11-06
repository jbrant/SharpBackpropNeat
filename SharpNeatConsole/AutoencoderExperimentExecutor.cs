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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Xml;
using log4net.Config;
using SharpNeat.Core;
using SharpNeat.Domains.EvolvedAutoEncoderHyperNeat;
using SharpNeat.Genomes.Neat;

#endregion

namespace SharpNeatConsole
{
    /// <summary>
    ///     Minimal console application that hardwaires the setting up on a evolution algorithm and start it running.
    /// </summary>
    internal class AutoencoderExperimentExecutor
    {
        private static IGenomeFactory<NeatGenome> _genomeFactory;
        private static List<NeatGenome> _genomeList;
        private static INeatEvolutionAlgorithm<NeatGenome> _ea;
        private static NumberFormatInfo _filenameNumberFormatter;
        private static string _logFilesBaseName;
        private static EvolvedAutoEncoderHyperNeatExperiment _autoencoderExperiment;
        private static XmlWriterSettings _xwSettings;
        private static StreamWriter _logFileWriter;

        private static void Main(string[] args)
        {
            Debug.Assert(args != null && args.Length == 5,
                "Invocation of the form: sharpneatconsole {experiment configuration file} {population seed file} {number of runs} {max generations} {log files base name}");

            // Read in experiment configuration file
            string experimentConfigurationFile = args[0];
            string seedPopulationFile = args[1];
            int numRuns = Int32.Parse(args[2]);
            int maxGenerations = Int32.Parse(args[3]);
            _logFilesBaseName = args[4];

            // Initialise log4net (log to console).
            XmlConfigurator.Configure(new FileInfo("log4net.properties"));

            // Configure filename formatter
            _filenameNumberFormatter = new NumberFormatInfo();
            _filenameNumberFormatter.NumberDecimalSeparator = ",";

            // Configure XML writer
            _xwSettings = new XmlWriterSettings();
            _xwSettings.Indent = true;
            
            // Initialize experiment
            _autoencoderExperiment = new EvolvedAutoEncoderHyperNeatExperiment();

            // Load experiment configuration
            XmlDocument experimentConfig = new XmlDocument();
            experimentConfig.Load(experimentConfigurationFile);
            _autoencoderExperiment.Initialize("Evolved Autoencoder", experimentConfig.DocumentElement);

            for (int curRun = 0; curRun < numRuns; curRun++)
            {
                _logFilesBaseName += "_Run" + (curRun + 1).ToString() + '_';

                // Confiure log file writer
                string logFilename = _logFilesBaseName + '_' + DateTime.Now.ToString("yyyyMMdd") + ".log";
                _logFileWriter = new StreamWriter(logFilename, true);
                _logFileWriter.WriteLine(
                    "ClockTime,Gen,BestFitness,MeanFitness,MeanSpecieChampFitness,ChampComplexity,MeanComplexity,MaxComplexity,TotalEvaluationCount,EvaluationsPerSec,SearchMode");

                // Open and load population XML file.
                using (XmlReader xr = XmlReader.Create(seedPopulationFile))
                {
                    _genomeList = _autoencoderExperiment.LoadPopulation(xr);
                }
                _genomeFactory = _genomeList[0].GenomeFactory;
                Console.WriteLine("Loaded [{0}] genomes.", _genomeList.Count);

                // Create evolution algorithm and attach update event.
                _ea = _autoencoderExperiment.CreateEvolutionAlgorithm(_genomeFactory, _genomeList);
                _ea.UpdateEvent += ea_UpdateEvent;

                // Start algorithm (it will run on a background thread).
                _ea.StartContinue();

                while (RunState.Terminated != _ea.RunState && RunState.Paused != _ea.RunState &&
                       _ea.CurrentGeneration < maxGenerations)
                {
                    Thread.Sleep(1000);
                }
            }
        }

        private static void ea_UpdateEvent(object sender, EventArgs e)
        {
            Console.WriteLine("gen={0:N0} bestFitness={1:N6}", _ea.CurrentGeneration, _ea.Statistics._maxFitness);

            // Derive best genome filename
            string bestGenomeFile = string.Format(_filenameNumberFormatter, "{0}_{1:0.00}_{2:yyyyMMdd_HHmmss}.gnm.xml",
                _logFilesBaseName, _ea.CurrentChampGenome.EvaluationInfo.Fitness, DateTime.Now);

            // Save genome to xml file.
            using (XmlWriter xw = XmlWriter.Create(bestGenomeFile, _xwSettings))
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
        }
    }
}