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

namespace SharpNeat.Phenomes
{
    /// <summary>
    ///     IBlackBox represents an abstract device, system or function which has inputs and outputs. The internal
    ///     workings and state of the box are not relevant to any method or class that acceps an IBlackBox - only that it
    ///     has inputs and outputs and a means of activation. In NEAT the neural network implementations generally fit this
    ///     pattern, that is:
    ///     - inputs are fed to a network.
    ///     - The network is actvated (e.g. some fixed number of timesteps or to relaxation).
    ///     - The network outputs are read and fed into the evaluation/scoring/fitness scheme.
    ///     From wikipedia:
    ///     Black box is a technical term for a device or system or object when it is viewed primarily in terms
    ///     of its input and output characteristics. Almost anything might occasionally be referred to as a black box -
    ///     a transistor, an algorithm, humans, the Internet.
    /// </summary>
    public interface IBlackBox
    {
        /// <summary>
        ///     Gets the number of inputs to the blackbox. This is assumed to be fixed for the lifetime of the IBlackBox.
        /// </summary>
        int InputCount { get; }

        /// <summary>
        ///     Gets the number of outputs from the blackbox. This is assumed to be fixed for the lifetime of the IBlackBox.
        /// </summary>
        int OutputCount { get; }

        /// <summary>
        ///     Gets an array of input values that feed into the black box.
        /// </summary>
        ISignalArray InputSignalArray { get; }

        /// <summary>
        ///     Gets an array of output values that feed out from the black box.
        /// </summary>
        ISignalArray OutputSignalArray { get; }

        /// <summary>
        ///     Gets a value indicating whether the black box's internal state is valid. It may become invalid if e.g. we ask a
        ///     recurrent
        ///     neural network to relax and it is unable to do so.
        /// </summary>
        bool IsStateValid { get; }

        /// <summary>
        ///     Activate the black box. This is a request for the box to accept its inputs and produce output signals
        ///     ready for reading from OutputSignalArray.
        /// </summary>
        void Activate();

        /// <summary>
        ///     Calculates the network error given some real-valued target and learning rate, using a back-propagation variant.
        /// </summary>
        /// <param name="targets">
        ///     The target values against which to calculate the error (note that it is assumed that the order of
        ///     target outputs matches the order of network outputs.
        /// </param>
        /// <param name="learningRate">The learning rate for the weight updates.</param>
        /// <returns>The overall error of the network.</returns>
        double CalculateError(double[] targets, double learningRate);

        /// <summary>
        ///     Calculates the network error given some learning rate using a back-propagation variant.  Note that this is
        ///     primarily for use with autoencoders as the input nodes are used as the targets, thus attempting to learn the
        ///     identity function.
        /// </summary>
        /// <param name="learningRate">The learning rate for the weight updates.</param>
        /// <returns>The overall error of the network.</returns>
        double CalculateError(double learningRate);

        /// <summary>
        ///     Reset any internal state.
        /// </summary>
        void ResetState();
    }
}