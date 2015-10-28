#region

using System.Collections.Generic;
using System.Linq;
using SharpNeat.Core;
using SharpNeat.Phenomes;
using SharpNeat.Utility;
using System;

#endregion

namespace SharpNeat.Domains.EvolvedAutoencoder
{
    public class EvolvedAutoencoderEvaluator : IPhenomeEvaluator<IBlackBox, FitnessInfo>
    {
        #region Constructors

        /// <summary>
        ///     Evaluator constructor which reads in the images on which the network are to be trained/evaluated.
        /// </summary>
        /// <param name="trainingImagesPath">The file name containing the training image samples.</param>
        /// <param name="imageResolution">The number of pixels constituting the resolution of the image(s).</param>
        /// <param name="numImageSamples">The number of sample training images in the given file.</param>
        /// <param name="learningRate">The learning rate for backpropagation.</param>
        /// <param name="numBackpropIterations">The target error to reach when running backpropagation.</param>
        /// <param name="trainingSampleProportion">
        ///     The proportion of the sample dataset to use for training (the result will be
        ///     used as part of the validation dataset).
        /// </param>        
        /// <param name="reduceAmountPerSide">The amount to divide each side by.</param>
        public EvolvedAutoencoderEvaluator(string trainingImagesPath, int imageResolution, int numImageSamples,
            double learningRate, int numBackpropIterations, double trainingSampleProportion, int reduceAmountPerSide)
        {            
            // Read in the images on which the network will be trained
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, imageResolution, numImageSamples,
                255);
            allImageSamples = ImageIoUtils.ReduceImages(allImageSamples, reduceAmountPerSide, (int)System.Math.Sqrt(imageResolution));
            // Determine the ending index of the training sample
            int trainingSampleEndIndex = (int) (allImageSamples.Count*trainingSampleProportion) - 1;

            // Extract the training and validation sample images
            _trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            _validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();

            // Set the learning rate
            _learningRate = learningRate;

            // Set the number of backpropagation iterations
            _numBackpropIterations = numBackpropIterations;

            // The maximum fitness ends up being the product of the number of validation samples 
            // and the number of input nodes (i.e. the image resolution)
            _maxFitness = _validationImageSamples.Count*imageResolution / (reduceAmountPerSide * reduceAmountPerSide);
        }

        #endregion

        #region Private instance fields

        private readonly List<double[]> _trainingImageSamples;
        private readonly List<double[]> _validationImageSamples;
        private readonly double _learningRate;
        private readonly int _numBackpropIterations;
        private readonly int _maxFitness;

        #endregion

        #region Evaluator Properties

        /// <summary>
        ///     The total number of evaluations that have been performed.
        /// </summary>
        public ulong EvaluationCount { get; private set; }

        /// <summary>
        ///     Indicates whether some goal fitness has been achieved.  For the autoencoder domain, this might mean
        ///     training/evolution to some acceptable amount of error.
        /// </summary>
        public bool StopConditionSatisfied { get; private set; }

        #endregion

        #region Evaluator methods

        /// <summary>
        /// the wrapper function that calls the actual Evaluate method
        /// </summary>
        /// <param name="phenome"></param>
        /// <returns></returns>
        public FitnessInfo Evaluate(IBlackBox phenome)
        {
            return EvaluateBasedOnNumOfBP(phenome);
        }

        /// <summary>
        /// Run BP until the score of the phenome exceeds a certain value. 
        /// Fitness is largely dictated by how little BP is needed, rather then getting a perfect Autoencoder.
        /// </summary>
        /// <param name="phenome"></param>
        /// <returns></returns>
        public FitnessInfo EvaluateBasedOnNumOfBP(IBlackBox phenome)
        {
            EvaluationCount++;
            double fitness = 0;
            double errorSum = 0;
            #region Backpropagate error while checking if error is acceptable(if so, give fitness base on the number of iterations used)
            for (int iter = 0; iter < _numBackpropIterations; iter++)
            {
                // Evaluate on each training sample
                foreach (double[] trainingImageSample in _trainingImageSamples)
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
                    phenome.CalculateError(_learningRate);
                }

                errorSum = 0;
                // Now we're going to validate how well the network performs on the validation set
                foreach (double[] validationImageSample in _validationImageSamples)
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

                if ((_maxFitness - errorSum) / _maxFitness * 100 > 99.5f)
                {
                    fitness = (_numBackpropIterations - iter) * 100f + ((_maxFitness - errorSum) / _maxFitness * 100 - 99f) * 100;
                    return new FitnessInfo(fitness, fitness);
                }
            }
            #endregion

            #region  Now we're going to validate how well the network performs on the validation set
            errorSum = 0;
            //Must be here in case _numBackpropIterations = 0
            foreach (double[] validationImageSample in _validationImageSamples)
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
            #endregion

            // Calculate the fitness as the difference between the maximum possible fitness 
            // and the sum of the errors on the validation set
            //double fitness = Math.Max(0, ((_maxFitness - errorSum)/ _maxFitness * 100))*10;
            fitness = Math.Max(0, (_maxFitness - errorSum) / _maxFitness * 100);
            return new FitnessInfo(fitness, fitness);
        }

        /// <summary>
        /// Run BP a fixed number of times, and then run the validation set.  
        /// Solutions may be harder to find, as if BP is run a lot even bad solutions can seem good.
        ///  If it's run a minimal amount of times, they may all find bad results.
        /// </summary>
        /// <param name="phenome"></param>
        /// <returns></returns>
        public FitnessInfo EvaluateBaedOnBestAutoencoder(IBlackBox phenome)
        {
            EvaluationCount++;
            double fitness = 0;
            double errorSum = 0;
            #region Backpropagate error for the specified number of iterations
            for (int iter = 0; iter < _numBackpropIterations; iter++)
            {
                errorSum = 0;
                // Evaluate on each training sample
                foreach (double[] trainingImageSample in _trainingImageSamples)
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
                    //errorSum += phenome.CalculateError(_learningRate);
                }
                double maxFitTrain = _trainingImageSamples.Count * _trainingImageSamples[0].Length;
                fitness = (maxFitTrain - errorSum) / maxFitTrain * 100;
            }
            #endregion

            #region  Now we're going to validate how well the network performs on the validation set
            errorSum = 0;
            foreach (double[] validationImageSample in _validationImageSamples)
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
            #endregion

            // Calculate the fitness as the difference between the maximum possible fitness 
            // and the sum of the errors on the validation set
            //double fitness = Math.Max(0, ((_maxFitness - errorSum)/ _maxFitness * 100))*10;
            fitness = Math.Max(0, (_maxFitness - errorSum) / _maxFitness * 100);
            return new FitnessInfo(fitness, fitness);
        }

        /// <summary>
        ///     Reset the internal state of the evaluation scheme if any exists.
        ///     Note: autoencoder evolution stores no internal state, so this method does nothing.
        /// </summary>
        public void Reset()
        {
        }

        #endregion
    }
}