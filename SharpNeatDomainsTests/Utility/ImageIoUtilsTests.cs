#region

using System.Collections.Generic;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#endregion

namespace SharpNeat.Utility.Tests
{
    [TestClass]
    public class ImageIoUtilsTests
    {
        [TestMethod]
        public void ReadImageTest()
        {
            string workingDirectory = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory()));

            // Read in data for the number 0
            List<double[]> num0Data =
                ImageIoUtils.ReadImage(
                    workingDirectory + "/../SharpNeatDomains/EvolvedAutoencoder/ImageData/Number0Samples.data", 28*28,
                    1000, 255);

            // Convert the first sample to a bitmap
            ImageIoUtils.WriteImage(workingDirectory + "/Number0.bmp", num0Data[0]);
        }
    }
}