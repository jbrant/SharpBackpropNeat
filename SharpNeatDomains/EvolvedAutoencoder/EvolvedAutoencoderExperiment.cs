#region

using System.Collections.Generic;
using System.Threading.Tasks;
using System.Xml;
using SharpNeat.Core;
using SharpNeat.Decoders;
using SharpNeat.Decoders.Neat;
using SharpNeat.DistanceMetrics;
using SharpNeat.EvolutionAlgorithms;
using SharpNeat.EvolutionAlgorithms.ComplexityRegulation;
using SharpNeat.Genomes.AutoencoderNeat;
using SharpNeat.Genomes.Neat;
using SharpNeat.Loggers;
using SharpNeat.Network;
using SharpNeat.Phenomes;
using SharpNeat.SpeciationStrategies;

#endregion

namespace SharpNeat.Domains.EvolvedAutoencoder
{
    public class EvolvedAutoencoderExperiment : IGuiNeatExperiment
    {
        #region Private instance variables

        private NetworkActivationScheme _activationScheme;
        private string _complexityRegulationStr;
        private int? _complexityThreshold;
        private string _generationalLogFile;
        private ParallelOptions _parallelOptions;
        private string _trainingImagesFilename;
        private int _numImageSamples;
        private double _learningRate;
        private double _backpropagationError;
        private double _trainingSampleProportion;

        #endregion

        #region Experiment Properties

        /// <summary>
        ///     The name of the experiment.
        /// </summary>
        public string Name { get; protected set; }

        /// <summary>
        ///     The description of the experiment.
        /// </summary>
        public string Description { get; protected set; }

        /// <summary>
        ///     The number of inputs to the neural network.
        /// </summary>
        public int InputCount { get; protected set; }

        /// <summary>
        ///     The number of outputs from the neural network (for an autoencoder, this should obviously match the number of
        ///     inputs).
        /// </summary>
        public int OutputCount { get; protected set; }

        /// <summary>
        ///     The size of the population.
        /// </summary>
        public int DefaultPopulationSize { get; protected set; }

        /// <summary>
        ///     The set of parameters with which to run the NEAT algorithm.
        /// </summary>
        public NeatEvolutionAlgorithmParameters NeatEvolutionAlgorithmParameters { get; protected set; }

        /// <summary>
        ///     The set of parameters controlling the specifics of genome evolution.
        /// </summary>
        public NeatGenomeParameters NeatGenomeParameters { get; protected set; }

        #endregion

        #region Experiment public methods

        /// <summary>
        ///     Initializes the experiment with some optional XML configutation data.
        /// </summary>
        /// <param name="name">The name of the experiment.</param>
        /// <param name="xmlConfig">The reference to the top-level configuration element for the experiment.</param>
        public void Initialize(string name, XmlElement xmlConfig)
        {
            // Read in boiler plate configuration settings
            Name = name;
            Description = XmlUtils.TryGetValueAsString(xmlConfig, "Description");
            DefaultPopulationSize = XmlUtils.GetValueAsInt(xmlConfig, "PopulationSize");
            InputCount = OutputCount = XmlUtils.GetValueAsInt(xmlConfig, "AutoencoderSize");

            // Read in algorithm/logging configuration
            _activationScheme = ExperimentUtils.CreateActivationScheme(xmlConfig, "Activation");
            _complexityRegulationStr = XmlUtils.TryGetValueAsString(xmlConfig, "ComplexityRegulationStrategy");
            _complexityThreshold = XmlUtils.TryGetValueAsInt(xmlConfig, "ComplexityThreshold");
            _parallelOptions = ExperimentUtils.ReadParallelOptions(xmlConfig);
            _generationalLogFile = XmlUtils.TryGetValueAsString(xmlConfig, "GenerationalLogFile");

            // Construct NEAT EA parameters
            NeatEvolutionAlgorithmParameters = new NeatEvolutionAlgorithmParameters();
            NeatEvolutionAlgorithmParameters.SpecieCount = XmlUtils.GetValueAsInt(xmlConfig, "SpecieCount");

            // Construct NEAT genome parameters
            NeatGenomeParameters = new NeatGenomeParameters();
            NeatGenomeParameters.FeedforwardOnly = _activationScheme.AcyclicNetwork;
            NeatGenomeParameters.ActivationFn = PlainSigmoid.__DefaultInstance;

            // Read in experiment domain-specific parameters
            _trainingImagesFilename = XmlUtils.TryGetValueAsString(xmlConfig, "TrainingImages");
            _numImageSamples = XmlUtils.GetValueAsInt(xmlConfig, "NumImageSamples");
            _learningRate = XmlUtils.GetValueAsDouble(xmlConfig, "LearningRate");
            _backpropagationError = XmlUtils.GetValueAsDouble(xmlConfig, "BackpropagationError");
            _trainingSampleProportion = XmlUtils.GetValueAsDouble(xmlConfig, "TrainingSampleProportion");
        }

        /// <summary>
        ///     Loads a population of genomes from an XmlReader and returns the genomes in a new list.  The genome factory for the
        ///     genomes can be obtained from any one of the genomes.
        /// </summary>
        /// <param name="xr">The XML reader to use for reading in the population.</param>
        /// <returns></returns>
        public List<NeatGenome> LoadPopulation(XmlReader xr)
        {
            NeatGenomeFactory genomeFactory = (NeatGenomeFactory) CreateGenomeFactory();
            return NeatGenomeXmlIO.ReadCompleteGenomeList(xr, false, genomeFactory);
        }

        /// <summary>
        ///     Saves a population of genomes to an XmlWriter.
        /// </summary>
        /// <param name="xw">The XML writer to serialize the population of genomes.</param>
        /// <param name="genomeList">The list of genomes to serialize.</param>
        public void SavePopulation(XmlWriter xw, IList<NeatGenome> genomeList)
        {
            NeatGenomeXmlIO.WriteComplete(xw, genomeList, false);
        }

        /// <summary>
        ///     Creates a genome decoder for the experiment.
        /// </summary>
        /// <returns>The NEAT genome decoder.</returns>
        public IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder()
        {
            return new NeatGenomeDecoder(_activationScheme);
        }

        /// <summary>
        ///     Creates a genome factory for the experiment with the given number of input and output nodes.
        /// </summary>
        /// <returns>The NEAT genome factory.</returns>
        public IGenomeFactory<NeatGenome> CreateGenomeFactory()
        {
            //return new NeatGenomeFactory(InputCount, OutputCount, NeatGenomeParameters);
            return new AutoencoderGenomeFactory(InputCount, OutputCount, 1, NeatGenomeParameters);
        }

        /// <summary>
        ///     Creates and returns a GenerationalNeatEvolutionAlgorithm object ready for running the NEAT algorithm/search.
        ///     Various sub-parts of the algorithm are also constructed and connected up.  Uses the experiments default population
        ///     size defined in the experiment's config XML.
        /// </summary>
        /// <returns>The NEAT evolution algorithm.</returns>
        public INeatEvolutionAlgorithm<NeatGenome> CreateEvolutionAlgorithm()
        {
            return CreateEvolutionAlgorithm(DefaultPopulationSize);
        }

        /// <summary>
        ///     Creates and returns a GenerationalNeatEvolutionAlgorithm object ready for running the NEAT algorithm/search.
        ///     Various sub-parts of the algorithm are also constructed and connected up.  This overload accepts a population size
        ///     parameter that specifies how many genomes to create in an initial randomly generated population.
        /// </summary>
        /// <param name="populationSize">The genome population size.</param>
        /// <returns>The NEAT evolution algorithm.</returns>
        public INeatEvolutionAlgorithm<NeatGenome> CreateEvolutionAlgorithm(int populationSize)
        {
            // Create a genome factory with our neat genome parameters object and the appropriate number of input and output neuron genes.
            IGenomeFactory<NeatGenome> genomeFactory = CreateGenomeFactory();

            // Create an initial population of randomly generated genomes.
            List<NeatGenome> genomeList = genomeFactory.CreateGenomeList(populationSize, 0);

            // Create evolution algorithm.
            return CreateEvolutionAlgorithm(genomeFactory, genomeList);
        }

        /// <summary>
        ///     Creates and returns a GenerationalNeatEvolutionAlgorithm object ready for running the NEAT algorithm/search.
        ///     Various sub-parts of the algorithm are also constructed and connected up.  This overload accepts a pre-built genome
        ///     population and their associated/parent genome factory.
        /// </summary>
        /// <param name="genomeFactory">The NEAT genome factory.</param>
        /// <param name="genomeList">The initial list of genomes.</param>
        /// <returns>The NEAT evolution algorithm.</returns>
        public INeatEvolutionAlgorithm<NeatGenome> CreateEvolutionAlgorithm(IGenomeFactory<NeatGenome> genomeFactory,
            List<NeatGenome> genomeList)
        {
            FileDataLogger logger = null;

            // Create distance metric. Mismatched genes have a fixed distance of 10; for matched genes the distance is their weigth difference
            IDistanceMetric distanceMetric = new ManhattanDistanceMetric(1.0, 0.0, 10.0);
            ISpeciationStrategy<NeatGenome> speciationStrategy =
                new ParallelKMeansClusteringStrategy<NeatGenome>(distanceMetric, _parallelOptions);

            // Create complexity regulation strategy
            IComplexityRegulationStrategy complexityRegulationStrategy =
                ExperimentUtils.CreateComplexityRegulationStrategy(_complexityRegulationStr, _complexityThreshold);

            // Initialize the logger
            if (_generationalLogFile != null)
            {
                logger =
                    new FileDataLogger(_generationalLogFile);
            }

            // Create the evolution algorithm
            GenerationalNeatEvolutionAlgorithm<NeatGenome> ea =
                new GenerationalNeatEvolutionAlgorithm<NeatGenome>(NeatEvolutionAlgorithmParameters, speciationStrategy,
                    complexityRegulationStrategy, logger);

            // Create evalutor
            EvolvedAutoencoderEvaluator evaluator = new EvolvedAutoencoderEvaluator(_trainingImagesFilename,
                InputCount, _numImageSamples, _learningRate, _backpropagationError, _trainingSampleProportion);

            // Create genome decoder
            IGenomeDecoder<NeatGenome, IBlackBox> genomeDecoder = CreateGenomeDecoder();

            // Create a genome list evaluator. This packages up the genome decoder with the genome evaluator
            IGenomeEvaluator<NeatGenome> innerFitnessEvaluator =
                new ParallelGenomeFitnessEvaluator<NeatGenome, IBlackBox>(genomeDecoder, evaluator, _parallelOptions);

            // Wrap the list evaluator in a 'selective' evaulator that will only evaluate new genomes. That is, we skip re-evaluating any genomes
            // that were in the population in previous generations (elite genomes). This is determined by examining each genome's evaluation info object.
            IGenomeEvaluator<NeatGenome> selectiveFitnessEvaluator = new SelectiveGenomeFitnessEvaluator<NeatGenome>(
                innerFitnessEvaluator,
                SelectiveGenomeFitnessEvaluator<NeatGenome>.CreatePredicate_OnceOnly());

            // Initialize the evolution algorithm
            ea.Initialize(selectiveFitnessEvaluator, genomeFactory, genomeList);

            // Finished. Return the evolution algorithm
            return ea;
        }

        /// <summary>
        ///     Creates a System.Windows.Forms derived object for displaying genomes.
        /// </summary>
        /// <returns>The windows form in which to display the phenotypic representation of the best performing genome.</returns>
        public AbstractGenomeView CreateGenomeView()
        {
            return new NeatGenomeView();
        }

        /// <summary>
        ///     Creates a System.Windows.Forms derived object for displaying output for a domain (e.g. show best genome's
        ///     output/performance/behaviour in the domain).
        /// </summary>
        /// <returns>The windows form in which to display the domain performance.</returns>
        public AbstractDomainView CreateDomainView()
        {
            return null;
        }

        #endregion
    }
}