#region

using System.Collections.Generic;
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
        public EvolvedAutoencoderEvaluator(string trainingImagesPath, int imageResolution, int numImageSamples,
            double learningRate)
        {
            // TODO: This could be split into training/validation sets

            // Read in the images on which the network will be trained
            _trainingImageSamples = ImageIoUtils.ReadImage(trainingImagesPath, imageResolution, numImageSamples);

            // Set the learning rate
            _learningRate = learningRate;
        }

        #endregion

        #region Private instance fields

        private readonly List<double[]> _trainingImageSamples;
        private readonly double _learningRate;

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

            // TODO: Probably need to choose some subset of images to train on instead of just going through first dozen

            for (int sampleIdx = 0; sampleIdx < 1; sampleIdx++)
            {
                // Reset the network
                phenome.ResetState();

                // Get the current sample image
                double[] curSample = _trainingImageSamples[sampleIdx];

                // Load the network inputs
                for (int pixelIdx = 0; pixelIdx < curSample.Length; pixelIdx++)
                {
                    phenome.InputSignalArray[pixelIdx] = curSample[pixelIdx];
                }

                // After inputs have been loaded, activate the network
                phenome.Activate();

                double curError;
                do
                {
                    curError = phenome.CalculateError(_learningRate);
                } while (curError > 0.1);
            }

            // TODO: After this, we present the validation set to the autoencoder and its score is stored in a FitnessInfo and returned

            return new FitnessInfo(0, 0);
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