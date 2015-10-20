#region

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using SharpNeat.Phenomes;

#endregion

namespace SharpNeat.Utility
{
    /// <summary>
    ///     The Image IO utils contains static methods for reading and writing binary/bitmap images.
    /// </summary>
    public static class ImageIoUtils
    {
        /// <summary>
        ///     Reads the binary image data for the specified number of image samples from the given file at the specified
        ///     resolution.
        /// </summary>
        /// <param name="imagePath">The path to the binary image file.</param>
        /// <param name="imageResolution">The resolution of the image.</param>
        /// <param name="numSamples">
        ///     The number of image samples in the file (this is essentially the distinct number of training
        ///     pictures).
        /// </param>
        /// <param name="pixelIntensityRange">The max numerical range of the pixel light intensity (e.g. 255 for grayscale).</param>
        /// <returns></returns>
        public static List<double[]> ReadImage(string imagePath, int imageResolution, int numSamples,
            int pixelIntensityRange)
        {
            List<double[]> imageData = new List<double[]>(numSamples);

            int sideLength = (int)Math.Sqrt(imageResolution);
            // Initialize the binary reader pointing at the image file
            BinaryReader reader = new BinaryReader(File.OpenRead(imagePath));

            // Iterate through every sample and extract its corresponding pixels
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
            {
                // Initialize new array for sample image pixel values
                double[] imagePixels = new double[imageResolution];

                // Extract each pixel in the current sample image
                for (int pixelIdx = 0; pixelIdx < imageResolution; pixelIdx++)
                {
                    imagePixels[pixelIdx] = reader.ReadByte() / (double)pixelIntensityRange;
                }                
                
                // Add image pixels to list
                imageData.Add(imagePixels);
            }

            return imageData;
        }

        public static void WriteImage(string imageName, ISignalArray outputSignalArray)
        {
            double[] convertedSignalArray = new double[outputSignalArray.Length];

            for (int idx = 0; idx < outputSignalArray.Length; idx++)
            {
                convertedSignalArray[idx] = outputSignalArray[idx];
            }

            WriteImage(imageName, convertedSignalArray);
        }

        /// <summary>
        ///     Converts the given double array image data to a grayscale bitmap and writes it to the specified file.
        /// </summary>
        /// <param name="imageName">The name of the image file.</param>
        /// <param name="imageData">The grayscale pixel intensities at each coordinate in the image canvas.</param>
        public static void WriteImage(string imageName, double[] imageData)
        {
            // For the MNIST dataset, this is a square, so the side lengths are all equal
            int sideLength = (int)Math.Sqrt(imageData.Length);

            // Create a square bitmap
            Bitmap imageBitmap = new Bitmap(sideLength, sideLength);

            for (int heightIdx = 0; heightIdx < sideLength; heightIdx++)
            {
                for (int widthIdx = 0; widthIdx < sideLength; widthIdx++)
                {
                    // Get the numeric intensity of the grayscale pixel
                    int numericColor = (int)(imageData[heightIdx * sideLength + widthIdx] * 255);

                    // Create the color (grayscale) component
                    Color pixelColor = Color.FromArgb(numericColor, numericColor, numericColor);

                    // Set the pixel at the given 2-dimensional location with the derived color
                    imageBitmap.SetPixel(widthIdx, heightIdx, pixelColor);
                }
            }

            // Save the image
            imageBitmap.Save(imageName);
        }

        #region Reduce Image Resolutions    
        /// <summary>
        ///     Reduces the given image by @reduceAmountPerSide
        ///     @reduceAmountPerSide MUST DIVIDE EVENLY INTO @sideLength
        /// </summary>
        /// <param name="imagePixels">The values of the image.</param>
        /// <param name="reduceAmountPerSide">The amount to divide each side by.</param>
        /// <param name="sideLength">The num of pixels on the side of the image(IE 11 by 11 means sideLength = 11).</param>
        public static double[] ReduceImage(double[] imagePixels, int reduceAmountPerSide, int sideLength)
        {
            double[] imagePixelsReduced = new double[sideLength * sideLength / (reduceAmountPerSide * reduceAmountPerSide)];
            // Create a square bitmap

            for (int heightIdx = 0; heightIdx < sideLength; heightIdx++)
            {
                for (int widthIdx = 0; widthIdx < sideLength; widthIdx++)
                {
                    int indLarge = widthIdx + heightIdx * sideLength;
                    int indexSmall = ((int)widthIdx) / reduceAmountPerSide + (int)(((int)(heightIdx / reduceAmountPerSide) * sideLength) / reduceAmountPerSide);
                    imagePixelsReduced[indexSmall] += imagePixels[indLarge] / (reduceAmountPerSide * reduceAmountPerSide);
                }
            }

            return imagePixelsReduced;
        }

        /// <summary>
        ///     Reduces all the given images by @reduceAmountPerSide
        ///     @reduceAmountPerSide MUST DIVIDE EVENLY INTO @sideLength
        /// </summary>
        /// <param name="imagePixels">The values of the images.</param>
        /// <param name="reduceAmountPerSide">The amount to divide each side by.</param>
        /// <param name="sideLength">The num of pixels on the side of the image(IE 11 by 11 means sideLength = 11).</param>
        public static List<double[]> ReduceImages(List<double[]> imagePixels, int reduceAmountPerSide, int sideLength)
        {
            List<double[]> imageData = new List<double[]>(imagePixels.Count);

            foreach (double[] data in imagePixels)
            {
                imageData.Add(ReduceImage(data, reduceAmountPerSide, sideLength));
            }

            return imageData;
        }
    }   
    #endregion
}
