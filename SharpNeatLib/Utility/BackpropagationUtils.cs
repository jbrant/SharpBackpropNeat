#region

using System;
using System.Diagnostics;
using System.Linq;
using SharpNeat.Network;
using SharpNeat.Phenomes;
using SharpNeat.Phenomes.NeuralNets;

#endregion

namespace SharpNeat.Utility
{
    /// <summary>
    ///     Contains utility methods for traditional backpropagation calculations (node errors, weight updates, and overall
    ///     network error).
    /// </summary>
    public static class BackpropagationUtils
    {
        /// <summary>
        ///     Calculates the output error for each node in the target layer and all hidden layers.  Note that this is a wrapper
        ///     method which takes a signal array, converts it to a double array, and passes that on to the method below.
        /// </summary>
        /// <param name="layers">The discrete layers in the ANN.</param>
        /// <param name="connections">Array of all connections in the ANN.</param>
        /// <param name="nodeActivationValues">The neuron activation values resulting from the last forward pass.</param>
        /// <param name="targetValues">The target values against which the network is being trained.</param>
        /// <param name="nodeActivationFunctions">The activation function for each neuron (this will only differ with HyperNEAT).</param>
        /// <returns>The errors for each output and hidden neuron.</returns>
        public static double[] CalculateErrorSignals(LayerInfo[] layers, FastConnection[] connections,
            double[] nodeActivationValues, ISignalArray targetValues, IActivationFunction[] nodeActivationFunctions)
        {
            double[] targets = new double[targetValues.Length];

            // Load the target double array from the input signal array
            targetValues.CopyTo(targets, 0);

            // Return the error signals
            return CalculateErrorSignals(layers, connections, nodeActivationValues, targets, nodeActivationFunctions);
        }

        /// <summary>
        ///     Calculates the output error for each node in the target layer and all hidden layers.
        /// </summary>
        /// <param name="layers">The discrete layers in the ANN.</param>
        /// <param name="connections">Array of all connections in the ANN.</param>
        /// <param name="nodeActivationValues">The neuron activation values resulting from the last forward pass.</param>
        /// <param name="targetValues">The target values against which the network is being trained.</param>
        /// <param name="nodeActivationFunctions">The activation function for each neuron (this will only differ with HyperNEAT).</param>
        /// <returns>The errors for each output and hidden neuron.</returns>
        public static double[] CalculateErrorSignals(LayerInfo[] layers, FastConnection[] connections,
            double[] nodeActivationValues, double[] targetValues, IActivationFunction[] nodeActivationFunctions)
        {
            double[] signalErrors = new double[nodeActivationValues.Length];

            // Get the last connection
            int conIdx = connections.Length - 1;

            // Get the last of the output nodes
            int nodeIdx = nodeActivationValues.Length - 1;

            // Iterate through the layers in reverse, calculating the signal errors
            for (int layerIdx = layers.Length - 1; layerIdx > 0; layerIdx--)
            {
                // Handle the output layer as a special case, calculating the error against the given target
                if (layerIdx == layers.Length - 1)
                {
                    // Calculate the error for every output node with respect to its corresponding target value
                    for (; nodeIdx >= layers[layerIdx - 1]._endNodeIdx; nodeIdx--)
                    {
                        signalErrors[nodeIdx] =
                            (targetValues[(targetValues.Length - 1) - ((nodeActivationValues.Length - 1) - nodeIdx)] -
                             nodeActivationValues[nodeIdx])*
                            nodeActivationFunctions[nodeIdx].CalculateDerivative(
                                nodeActivationValues[nodeIdx]);
                    }
                }

                // Otherwise, we're on a hidden layer, so just compute the error with respect to the target
                // node's error in the layer above
                else
                {
                    // Calculate the error for each hidden node with respect to the error of the 
                    // target node(s) of the next layer
                    for (; nodeIdx >= layers[layerIdx - 1]._endNodeIdx; nodeIdx--)
                    {
                        double deltas = 0;

                        // Calculate the sum of the products of the target node error and connection weight
                        while (connections[conIdx]._srcNeuronIdx == nodeIdx)
                        {
                            deltas += connections[conIdx]._weight*
                                      signalErrors[connections[conIdx]._tgtNeuronIdx];
                            conIdx--;
                        }

                        // The output error for the hidden node is the then the sum of the errors 
                        // plus the derivative of the activation function with respect to the output
                        signalErrors[nodeIdx] = deltas*
                                                nodeActivationFunctions[nodeIdx].CalculateDerivative(
                                                    nodeActivationValues[nodeIdx]);
                    }
                }
            }

            return signalErrors;
        }

        /// <summary>
        ///     Updates weights based on node error calculations using a given learning rate (momentum isn't taken into
        ///     consideration here).
        /// </summary>
        /// <param name="layers">The discrete layers in the ANN.</param>
        /// <param name="connections">Array of all connections in the ANN.</param>
        /// <param name="learningRate">The learning rate for all connections.</param>
        /// <param name="signalErrors">The errors for each output and hidden neuron.</param>
        /// <param name="nodeActivationValues">The activation function for each neuron (this will only differ with HyperNEAT).</param>
        public static void BackpropagateError(LayerInfo[] layers, FastConnection[] connections, double learningRate,
            double[] signalErrors, double[] nodeActivationValues)
        {
            int conIdx = 0;

            // Iterate through every layer in a forward pass, calculating the new weights on each connection
            for (int layerIdx = 1; layerIdx < layers.Length; layerIdx++)
            {
                // Start at one layer below the current layer so we can access the source nodes
                LayerInfo layerInfo = layers[layerIdx - 1];

                // Calculate the new weight for every connection in the current layer up to the last (i.e. "end") 
                // connection by adding its current weight to the product of the learning rate, target neuron error, 
                // and source neuron output
                for (; conIdx < layerInfo._endConnectionIdx; conIdx++)
                {
                    connections[conIdx]._weight = connections[conIdx]._weight +
                                                  learningRate*signalErrors[connections[conIdx]._tgtNeuronIdx]*
                                                  nodeActivationValues[connections[conIdx]._srcNeuronIdx];
                }
            }
        }

        /// <summary>
        ///     Calculates the overall network error based on the errors of the output and hidden nodes.  This is essentially doing
        ///     a residual sum of squares calculation (RSS).
        /// </summary>
        /// <param name="signalErrors">The errors for each output and hidden neuron.</param>
        /// <returns>The overall error (RSS) of the network.</returns>
        public static double CalculateOverallError(double[] signalErrors)
        {
            double totalError = signalErrors.Sum(signalError => Math.Pow(signalError, 2));

            return totalError/signalErrors.Length;
        }

        /// <summary>
        ///     Calculates the error of the network by comparing the difference between each input and its corresponding output.
        ///     The total of the differences is the error.
        /// </summary>
        /// <param name="inputSignalArray">The input signal array.</param>
        /// <param name="outputSignalArray">The output signal array.</param>
        /// <returns></returns>
        public static double CalculateOutputError(ISignalArray inputSignalArray, ISignalArray outputSignalArray)
        {
            // Make sure that the input and output array are of the same length
            Debug.Assert(inputSignalArray.Length == outputSignalArray.Length,
                "Input and output signal arrays are different lengths.");

            double activationDiff = 0.0;

            // Compare each input to its corresponding output, taking the absolute value of the difference in activation
            for (int idx = 0; idx < inputSignalArray.Length; idx++)
            {
                activationDiff += Math.Abs(inputSignalArray[idx] - outputSignalArray[idx]);
            }

            return activationDiff;
        }
    }
}