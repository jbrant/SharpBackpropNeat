#region

using System.Collections.Generic;
using System.Linq;
using SharpNeat.Core;
using SharpNeat.Phenomes;
using SharpNeat.Utility;

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
        /// <param name="maxTrainingError">The target error to reach when running backpropagation.</param>
        /// <param name="trainingSampleProportion">
        ///     The proportion of the sample dataset to use for training (the result will be
        ///     used as part of the validation dataset).
        /// </param>
        public EvolvedAutoencoderEvaluator(string trainingImagesPath, int imageResolution, int numImageSamples,
            double learningRate, double maxTrainingError, double trainingSampleProportion)
        {
            // TODO: This could be split into training/validation sets

            // Read in the images on which the network will be trained
            List<double[]> allImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, imageResolution, numImageSamples);

//            double[] sample1 = new[] {1.0, 0.0, 1.0, 0.0};
//            double[] sample2 = new[] {1.0, 1.0, 0.0, 1.0};
//            double[] sample3 = new[] { 1.0, 1.0, 1.0, 1.0 };
//            double[] sample4 = new[] { 1.0, 1.0, 1.0, 0.0 };
//            List<double[]> allImageSamples = new List<double[]>() { sample1, sample2, sample3, sample4 };

            // Determine the ending index of the training sample
            int trainingSampleEndIndex = (int) (allImageSamples.Count*trainingSampleProportion) - 1;

            // TODO: Need a more robust sampling method
            // Extract the training and validation sample images
            _trainingImageSamples = allImageSamples.GetRange(0, trainingSampleEndIndex);
            _validationImageSamples = allImageSamples.Skip(trainingSampleEndIndex + 1).ToList();

            // Set the learning rate
            _learningRate = learningRate;

            // Set desired maximum error
            _maxTrainingError = maxTrainingError;
        }

        #endregion

        #region Private instance fields

        private readonly List<double[]> _trainingImageSamples;
        private readonly List<double[]> _validationImageSamples;
        private readonly double _learningRate;
        private readonly double _maxTrainingError;

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

        #region Evaluator public methods

        public FitnessInfo Evaluate(IBlackBox phenome)
        {
            EvaluationCount++;

            // Evaluate on each training sample
            foreach (double[] trainingImageSample in _trainingImageSamples)
            {
                double curError;
                
                // Reset the network
                phenome.ResetState();

                // TODO: Need to normalize the inputs in [0,1] range before feeding to network

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < trainingImageSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = trainingImageSample[pixelIdx];
                }

                // TODO: Need to add maximum number of iterations on the exit condition in order to avoid infinite loop
                do
                {
                    // After inputs have been loaded, activate the network
                    phenome.Activate();

                    // Calculate the overall error based on how closely the outputs match the inputs
                    curError = phenome.CalculateError(_learningRate);
                } while (curError > _maxTrainingError); // Continue training until error is reduced to desired level
            }

            double errorSum = 0;

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
                errorSum += phenome.CalculateError(_learningRate);
            }

            // TODO: Need a better fitness function than this
            double fitness = 1000 - errorSum;

            // TODO: Need to define a stop condition

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