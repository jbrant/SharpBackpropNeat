#region

using System;
using System.Collections.Generic;
using SharpNeat.Genomes.Neat;
using SharpNeat.Network;
using SharpNeat.Utility;

#endregion

namespace SharpNeat.Genomes.AutoencoderNeat
{
    public class AutoencoderGenomeFactory : NeatGenomeFactory
    {
        #region Instance fields

        private int _hiddenNeuronCount;

        #endregion

        #region Constructors

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount)
            : base(inputNeuronCount, outputNeuronCount)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount,
            NeatGenomeParameters neatGenomeParams) : base(inputNeuronCount, outputNeuronCount, neatGenomeParams)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount,
            NeatGenomeParameters neatGenomeParams, UInt32IdGenerator genomeIdGenerator,
            UInt32IdGenerator innovationIdGenerator)
            : base(inputNeuronCount, outputNeuronCount, neatGenomeParams, genomeIdGenerator, innovationIdGenerator)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount,
            IActivationFunctionLibrary activationFnLibrary)
            : base(inputNeuronCount, outputNeuronCount, activationFnLibrary)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount,
            IActivationFunctionLibrary activationFnLibrary, NeatGenomeParameters neatGenomeParams)
            : base(inputNeuronCount, outputNeuronCount, activationFnLibrary, neatGenomeParams)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public AutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount, int hiddenNeuronCount,
            IActivationFunctionLibrary activationFnLibrary, NeatGenomeParameters neatGenomeParams,
            UInt32IdGenerator genomeIdGenerator, UInt32IdGenerator innovationIdGenerator)
            : base(
                inputNeuronCount, outputNeuronCount, activationFnLibrary, neatGenomeParams, genomeIdGenerator,
                innovationIdGenerator)
        {
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        #endregion

        #region Overridden Methods

        /// <summary>
        ///     Creates a list of randomly initialised genomes.
        /// </summary>
        /// <param name="length">The number of genomes to create.</param>
        /// <param name="birthGeneration">
        ///     The current evolution algorithm generation.
        ///     Assigned to the new genomes as their birth generation.
        /// </param>
        public override List<NeatGenome> CreateGenomeList(int length, uint birthGeneration)
        {
            List<NeatGenome> genomeList = new List<NeatGenome>(length);
            for (int i = 0; i < length; i++)
            {
                // We reset the innovation ID to zero so that all created genomes use the same 
                // innovation IDs for matching neurons and connections. This isn't a strict requirement but
                // throughout the SharpNeat code we attempt to use the same innovation ID for like structures
                // to improve the effectiveness of sexual reproduction.
                _innovationIdGenerator.Reset();
                genomeList.Add(CreateGenome(birthGeneration));
            }
            return genomeList;
        }

        /// <summary>
        ///     Creates a single randomly initialised genome.
        ///     A random set of connections are made form the input to the output neurons, the number of
        ///     connections made is based on the NeatGenomeParameters.InitialInterconnectionsProportion
        ///     which specifies the proportion of all posssible input-output connections to be made in
        ///     initial genomes.
        ///     The connections that are made are allocated innovation IDs in a consistent manner across
        ///     the initial population of genomes. To do this we allocate IDs sequentially to all possible
        ///     interconnections and then randomly select some proportion of connections for inclusion in the
        ///     genome. In addition, for this scheme to work the innovation ID generator must be reset to zero
        ///     prior to each call to CreateGenome(), and a test is made to ensure this is the case.
        ///     The consistent allocation of innovation IDs ensure that equivalent connections in different
        ///     genomes have the same innovation ID, and although this isn't strictly necessary it is
        ///     required for sexual reproduction to work effectively - like structures are detected by comparing
        ///     innovation IDs only.
        /// </summary>
        /// <param name="birthGeneration">
        ///     The current evolution algorithm generation.
        ///     Assigned to the new genome as its birth generation.
        /// </param>
        public override NeatGenome CreateGenome(uint birthGeneration)
        {
            NeuronGeneList neuronGeneList = new NeuronGeneList(_inputNeuronCount + _outputNeuronCount + _hiddenNeuronCount);
            NeuronGeneList inputNeuronGeneList = new NeuronGeneList(_inputNeuronCount); // includes single bias neuron.
            NeuronGeneList outputNeuronGeneList = new NeuronGeneList(_outputNeuronCount);
            NeuronGeneList hiddenNeuronGeneList = new NeuronGeneList(_hiddenNeuronCount);

            // Create a single bias neuron.
            uint biasNeuronId = _innovationIdGenerator.NextId;
            if (0 != biasNeuronId)
            {
                // The ID generator must be reset before calling this method so that all generated genomes use the
                // same innovation ID for matching neurons and structures.
                throw new SharpNeatException("IdGenerator must be reset before calling CreateGenome(uint)");
            }

            // Note. Genes within nGeneList must always be arranged according to the following layout plan.
            //   Bias - single neuron. Innovation ID = 0
            //   Input neurons.
            //   Output neurons.
            //   Hidden neurons.
            NeuronGene neuronGene = CreateNeuronGene(biasNeuronId, NodeType.Bias);
            inputNeuronGeneList.Add(neuronGene);
            neuronGeneList.Add(neuronGene);

            // Create input neuron genes.
            for (int i = 0; i < _inputNeuronCount; i++)
            {
                neuronGene = CreateNeuronGene(_innovationIdGenerator.NextId, NodeType.Input);
                inputNeuronGeneList.Add(neuronGene);
                neuronGeneList.Add(neuronGene);
            }

            // Create output neuron genes. 
            for (int i = 0; i < _outputNeuronCount; i++)
            {
                neuronGene = CreateNeuronGene(_innovationIdGenerator.NextId, NodeType.Output);
                outputNeuronGeneList.Add(neuronGene);
                neuronGeneList.Add(neuronGene);
            }

            // Create hidden neuron genes.
            for (int i = 0; i < _hiddenNeuronCount; i++)
            {
                neuronGene = CreateNeuronGene(_innovationIdGenerator.NextId, NodeType.Hidden);
                hiddenNeuronGeneList.Add(neuronGene);
                neuronGeneList.Add(neuronGene);
            }

            // Define all possible connections between the input and hidden neurons and the hidden and output neurons 
            // (fully interconnected with minimal hidden/feature layer).
            int srcCount = inputNeuronGeneList.Count;
            int tgtCount = outputNeuronGeneList.Count;
            int hdnCount = hiddenNeuronGeneList.Count;
            ConnectionDefinition[] srcHdnConnectionDefArr = new ConnectionDefinition[srcCount * hdnCount];
            ConnectionDefinition[] hdnTgtConnectionDefArr = new ConnectionDefinition[tgtCount * hdnCount];

            for (int hdnIdx = 0, i = 0; hdnIdx < hdnCount; hdnIdx++)
            {
                for (int srcIdx = 0; srcIdx < srcCount; srcIdx++)
                {
                    srcHdnConnectionDefArr[i++] = new ConnectionDefinition(_innovationIdGenerator.NextId, srcIdx, hdnIdx);
                }                
            }

            for (int hdnIdx = 0, i = 0; hdnIdx < hdnCount; hdnIdx++)
            {
                for (int tgtIdx = 0; tgtIdx < tgtCount; tgtIdx++)
                {
                    hdnTgtConnectionDefArr[i++] = new ConnectionDefinition(_innovationIdGenerator.NextId, hdnIdx, tgtIdx);
                }
            }

            // Shuffle the array of possible connections.
            Utilities.Shuffle(srcHdnConnectionDefArr, _rng);
            Utilities.Shuffle(hdnTgtConnectionDefArr, _rng);

            // Select connection definitions from the head of the list and convert them to real connections.
            // We want some proportion of all possible connections but at least one (Connectionless genomes are not allowed).
            int srcConnectionCount = (int) Utilities.ProbabilisticRound(
                srcHdnConnectionDefArr.Length*_neatGenomeParamsComplexifying.InitialInterconnectionsProportion,
                _rng);
            srcConnectionCount = Math.Max(1, srcConnectionCount);

            int tgtConnectionCount = (int)Utilities.ProbabilisticRound(
                hdnTgtConnectionDefArr.Length * _neatGenomeParamsComplexifying.InitialInterconnectionsProportion,
                _rng);
            tgtConnectionCount = Math.Max(1, tgtConnectionCount);

            // Create the connection gene list and populate it.
            ConnectionGeneList connectionGeneList = new ConnectionGeneList(srcConnectionCount + tgtConnectionCount);

            for (int i = 0; i < srcConnectionCount; i++)
            {
                ConnectionDefinition def = srcHdnConnectionDefArr[i];
                NeuronGene srcNeuronGene = inputNeuronGeneList[def._sourceNeuronIdx];
                NeuronGene tgtNeuronGene = hiddenNeuronGeneList[def._targetNeuronIdx];

                ConnectionGene cGene = new ConnectionGene(def._innovationId,
                    srcNeuronGene.InnovationId,
                    tgtNeuronGene.InnovationId,
                    GenerateRandomConnectionWeight());
                connectionGeneList.Add(cGene);

                // Register connection with endpoint neurons.
                srcNeuronGene.TargetNeurons.Add(cGene.TargetNodeId);
                tgtNeuronGene.SourceNeurons.Add(cGene.SourceNodeId);
            }

            for (int i = 0; i < tgtConnectionCount; i++)
            {
                ConnectionDefinition def = hdnTgtConnectionDefArr[i];
                NeuronGene srcNeuronGene = hiddenNeuronGeneList[def._sourceNeuronIdx];
                NeuronGene tgtNeuronGene = outputNeuronGeneList[def._targetNeuronIdx];

                ConnectionGene cGene = new ConnectionGene(def._innovationId,
                    srcNeuronGene.InnovationId,
                    tgtNeuronGene.InnovationId,
                    GenerateRandomConnectionWeight());
                connectionGeneList.Add(cGene);

                // Register connection with endpoint neurons.
                srcNeuronGene.TargetNeurons.Add(cGene.TargetNodeId);
                tgtNeuronGene.SourceNeurons.Add(cGene.SourceNodeId);
            }

            // Ensure connections are sorted.
            connectionGeneList.SortByInnovationId();

            // Create and return the completed genome object.
            return CreateGenome(_genomeIdGenerator.NextId, birthGeneration,
                neuronGeneList, connectionGeneList,
                _inputNeuronCount, _outputNeuronCount, false);
        }

        #endregion
    }
}