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

using SharpNeat.Genomes.Neat;
using SharpNeat.Network;
using SharpNeat.Utility;
using System;

namespace SharpNeat.Genomes.AutoencoderNeat
{
    /// <summary>
    /// A sub-class of NeatGenomeFactory for creating CPPN genomes.
    /// </summary>
    public class CppnAutoencoderGenomeFactory : NeatGenomeFactory
    {
        #region Constructors

        /// <summary>
        /// Constructs with default NeatGenomeParameters, ID generators initialized to zero and a
        /// default IActivationFunctionLibrary.
        /// </summary>
        public CppnAutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount)
            : base(inputNeuronCount, outputNeuronCount, DefaultActivationFunctionLibrary.CreateLibraryCppn())
        {
        }

        /// <summary>
        /// Constructs with default NeatGenomeParameters, ID generators initialized to zero and the
        /// provided IActivationFunctionLibrary.
        /// </summary>
        public CppnAutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount,
                                 IActivationFunctionLibrary activationFnLibrary)
            : base(inputNeuronCount, outputNeuronCount, activationFnLibrary)
        {
        }

        /// <summary>
        /// Constructs with the provided IActivationFunctionLibrary and NeatGenomeParameters.
        /// </summary>
        public CppnAutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount,
                                 IActivationFunctionLibrary activationFnLibrary,
                                 NeatGenomeParameters neatGenomeParams)
            : base(inputNeuronCount,outputNeuronCount, activationFnLibrary, neatGenomeParams)
        {
        }

        /// <summary>
        /// Constructs with the provided IActivationFunctionLibrary, NeatGenomeParameters and ID generators.
        /// </summary>
        public CppnAutoencoderGenomeFactory(int inputNeuronCount, int outputNeuronCount,
                                 IActivationFunctionLibrary activationFnLibrary,
                                 NeatGenomeParameters neatGenomeParams,
                                 UInt32IdGenerator genomeIdGenerator, UInt32IdGenerator innovationIdGenerator)
            : base(inputNeuronCount, outputNeuronCount, activationFnLibrary, neatGenomeParams, genomeIdGenerator, innovationIdGenerator)
        {
        }

        #endregion

        #region Public Methods [NeatGenome Specific / CPPN Overrides]

        /// <summary>
        /// Override that randomly assigns activation functions to neuron's from an activation function library
        /// based on each library item's selection probability.
        /// </summary>
        public override NeuronGene CreateNeuronGene(uint innovationId, NodeType neuronType)
        {
            int activationFnId;
            switch(neuronType)
            {
                case NodeType.Bias:
                case NodeType.Input:
                case NodeType.Output:
                {   // Use the ID of the first function. By convention this will be the Linear function but in actual 
                    // fact bias and input neurons don't use their activation function.
                    activationFnId = _activationFnLibrary.GetFunctionList()[0].Id;
                    break;
                }
                default:
                {
                    activationFnId = _activationFnLibrary.GetRandomFunction(_rng).Id;
                    break;
                }
            }

            return new NeuronGene(innovationId, neuronType, activationFnId);
        }

        /// <summary>
        ///     TODO
        /// </summary>
        /// <param name="birthGeneration">
        ///     The current evolution algorithm generation.
        ///     Assigned to the new genome as its birth generation.
        /// </param>
        public override NeatGenome CreateGenome(uint birthGeneration)
        {
            NeuronGeneList neuronGeneList = new NeuronGeneList(_inputNeuronCount + _outputNeuronCount);
            NeuronGeneList inputNeuronGeneList = new NeuronGeneList(_inputNeuronCount); // includes single bias neuron.
            NeuronGeneList outputNeuronGeneList = new NeuronGeneList(_outputNeuronCount);

            // Create a single bias neuron.
            uint biasNeuronId = _innovationIdGenerator.NextId;
            if (0 != biasNeuronId)
            {   // The ID generator must be reset before calling this method so that all generated genomes use the
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

            // Define all possible connections between the input and output neurons (fully interconnected).
            int srcCount = inputNeuronGeneList.Count;
            int tgtCount = outputNeuronGeneList.Count;
            ConnectionDefinition[] connectionDefArr = new ConnectionDefinition[srcCount * tgtCount];

            for (int srcIdx = 0, i = 0; srcIdx < srcCount; srcIdx++)
            {
                for (int tgtIdx = 0; tgtIdx < tgtCount; tgtIdx++)
                {
                    connectionDefArr[i++] = new ConnectionDefinition(_innovationIdGenerator.NextId, srcIdx, tgtIdx);
                }
            }

            // Shuffle the array of possible connections.
            Utilities.Shuffle(connectionDefArr, _rng);

            // Select connection definitions from the head of the list and convert them to real connections.
            // We want some proportion of all possible connections but at least one (Connectionless genomes are not allowed).
            int connectionCount = (int)Utilities.ProbabilisticRound(
                (double)connectionDefArr.Length * _neatGenomeParamsComplexifying.InitialInterconnectionsProportion,
                _rng);
            connectionCount = Math.Max(1, connectionCount);

            // Create the connection gene list and populate it.
            ConnectionGeneList connectionGeneList = new ConnectionGeneList(connectionCount);

            #region Add connection to bisas short connections                 
            NeuronGene srcNeuronGeneACBias = inputNeuronGeneList[0];
            if (!srcNeuronGeneACBias.TargetNeurons.Contains(outputNeuronGeneList[2].InnovationId))
            {
                NeuronGene tgtNeuronGeneAC = outputNeuronGeneList[2];
                ConnectionGene biasGene = new ConnectionGene(_innovationIdGenerator.NextId,
                                                            srcNeuronGeneACBias.InnovationId,
                                                            tgtNeuronGeneAC.InnovationId,
                                                            Math.Abs(GenerateRandomConnectionWeight()));
                connectionGeneList.Add(biasGene);

                // Register connection with endpoint neurons.
                srcNeuronGeneACBias.TargetNeurons.Add(biasGene.TargetNodeId);
                tgtNeuronGeneAC.SourceNeurons.Add(biasGene.SourceNodeId);
            }
            double conW = GenerateRandomConnectionWeight();
            for (int i = 5; i <= 6; i++)
            {    
                NeuronGene srcNeuronGeneAC = inputNeuronGeneList[i];
                if (!srcNeuronGeneAC.TargetNeurons.Contains(outputNeuronGeneList[2].InnovationId))
                {
                    NeuronGene tgtNeuronGeneAC = outputNeuronGeneList[2];
                    ConnectionGene biasGene = new ConnectionGene(_innovationIdGenerator.NextId,
                                                                srcNeuronGeneAC.InnovationId,
                                                                tgtNeuronGeneAC.InnovationId,
                                                                -Math.Abs(conW));
                    connectionGeneList.Add(biasGene);

                    // Register connection with endpoint neurons.
                    srcNeuronGeneAC.TargetNeurons.Add(biasGene.TargetNodeId);
                    tgtNeuronGeneAC.SourceNeurons.Add(biasGene.SourceNodeId);
                }
            }
            #endregion

            #region Add connection to bisas connection strength based on distance 
            for (int i = 5; i <= 6; i++)
            {
                NeuronGene srcNeuronGeneAC = inputNeuronGeneList[i];
                if (!srcNeuronGeneAC.TargetNeurons.Contains(outputNeuronGeneList[0].InnovationId))
                {
                    NeuronGene tgtNeuronGeneAC = outputNeuronGeneList[0];
                    ConnectionGene biasGene = new ConnectionGene(_innovationIdGenerator.NextId,
                                                                srcNeuronGeneAC.InnovationId,
                                                                tgtNeuronGeneAC.InnovationId,
                                                                Math.Abs(GenerateRandomConnectionWeight()));
                    connectionGeneList.Add(biasGene);

                    // Register connection with endpoint neurons.
                    srcNeuronGeneAC.TargetNeurons.Add(biasGene.TargetNodeId);
                    tgtNeuronGeneAC.SourceNeurons.Add(biasGene.SourceNodeId);
                }
            }
            #endregion
            for (int i = 0; i < connectionCount; i++)
            {
                ConnectionDefinition def = connectionDefArr[i];
                NeuronGene srcNeuronGene = inputNeuronGeneList[def._sourceNeuronIdx];
                NeuronGene tgtNeuronGene = outputNeuronGeneList[def._targetNeuronIdx];

                ConnectionGene cGene = new ConnectionGene(def._innovationId,
                                                        srcNeuronGene.InnovationId,
                                                        tgtNeuronGene.InnovationId,
                                                        GenerateRandomConnectionWeight());
                if (!srcNeuronGene.TargetNeurons.Contains(cGene.TargetNodeId))
                {
                    connectionGeneList.Add(cGene);

                    // Register connection with endpoint neurons.
                    srcNeuronGene.TargetNeurons.Add(cGene.TargetNodeId);
                    tgtNeuronGene.SourceNeurons.Add(cGene.SourceNodeId);
                }
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
