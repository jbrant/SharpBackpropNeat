#region

using SharpNeat.Core;
using SharpNeat.Phenomes;

#endregion

namespace SharpNeat.Domains.EvolvedAutoencoder
{
    public class EvolvedAutoencoderEvaluator : IPhenomeEvaluator<IBlackBox, FitnessInfo>
    {
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

            // TODO: Here, we would present the training set to the given autoencoder (the phenome) and traing to some error

            // TODO: After this, we present the validation set to the autoencoder and its score is stored in a FitnessInfo and returned

            return new FitnessInfo(0,0);
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