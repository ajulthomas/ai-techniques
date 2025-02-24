# import required libraries
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from scipy import stats
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans

# tidying up the current workspace
plt.close("all")
clear = lambda: os.system("cls" if os.name == "nt" else "clear")
clear()
np.random.seed(110)


colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5], [0.5, 0, 0.5]]

imgNames = ["water_coins", "jump", "tiger"]
SegCounts = [2, 3, 4, 5]


 # min-max normalization
 # write a function that will convert the image pixel values to the range [0-1]
def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    
    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(img, dtype="float")

    new_img = (img - min_val) / (max_val - min_val)
    return(new_img)


# write a function that will convert the image pixel values from 0-255 to 0-1
def im2double(im):
    im = np.asarray(im, dtype='float')
    if(im.max() > 1):
        im /= 255.0
    return im

for imgName in imgNames:
    for SegCount in SegCounts:
        path = f"./input/{imgName}.png"
        print(path)

        # loading image using matplotlib mpimg libraray
        img = mpimg.imread(path)
        print(
            f"Using Matplotlib Image Library: \n Image is of datatype {img.dtype} and size {img.shape}\n"
        )

        # loading image using pillow
        img = np.asarray(Image.open(path))
        print(
            f"Using Pillow Library: \n Image is of datatype {img.dtype} and size {img.shape}\n"
        )

        # define parameters
        nSegments = SegCount
        nPixels = img.shape[0] * img.shape[1]
        nColors = 3;
        maxIterations = 20

        # output path = Output/SegCount_segments/imgName/
        outputPath = f"./output/{SegCount}_segments/{imgName}/"
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        # save using Matplotlib image Library mpimg as '0.png'
        mpimg.imsave(outputPath + "0.png", img)

        # vectorize the image for easier loops
        #  Reshape pixels as a nPixels X nColors X 1 matrix-- 5 points
        pixels = img
        pixels = pixels.reshape(nPixels, nColors, 1)

        """
        Initialize pi (mixture proportion) vector and mu matrix
        (containing means of each distribution)
        Vector of probabilities for segments... 1 value for each segment.
        The idea behind image generation goes Like this...
        When the image was generated color was determined for each pixel
        by selecting a value from one of "n" normal distributions
        corresponding to the "n" color segments. Each value
        in this vector corresponds to the probability that a
        particular segment or normal distribution was chosen.

        Initial guess for pi's is 1/nSegments. Small amount of noise
        added to slightly perturb GMM coefficients from the initial guess

        """

        pi = 1/nSegments * (np.ones((nSegments,1), dtype='float'))

        for seg_ctr in range(len(pi)):
            increment = np.random.normal(0, .0001, 1)

            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
            else:
                pi[seg_ctr] = pi[seg_ctr]- increment

        """
        Similarly, the initial guess for the segment color means would be a
        perturbed version of [mu_R, mu_G, mu_B],
        where mu_R, mu_G, mu_B respectively denote the means of the
        R,G,B color channels in the image.
        mu is a nSegments X nCoLors matrix,(segLabeLs*255).np.asarray
        (in int format) where each matrix row denotes mean RGB color
        for a particular segment.

        Initialize mu to 1/nSegments*['ones' matrix (whose elements are all 1) of size nSegments X nColors]
        
        """
        
        mu = (1/nSegments) * np.ones((nSegments, nColors), dtype='float')

        #add noise to the initialization (but keep it unit)
        for seg_ctr in range(nSegments):
            increment= np.random.normal(0,.0001,1)
            for col_ctr in range(nColors):
                if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) - increment
                
        # EM-iterations begin here. Start with the initial (pi, mu) guesses

        mu_last_iter=mu
        pi_last_iter=pi

        for iteration in range(maxIterations):
            # Expectation step
            #  print(".join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration:',str(iteration+1), ' E-step']))
            print(f'Image: {imgName} nSegments: {nSegments} iteration: {iteration+1} E-step')

            # Weights that describe the Likelihood that pixel denoted by "pix_ctr" belongs to a color cluster "seg_ctr"
            # temporarily reinitialize all weights to 1 of appropriate size, before they are recomputed
            Ws = np.ones((nPixels, nSegments), dtype='float')

            # Logarithmic form of the Estep
            for pix_ctr in range(nPixels):
                # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float')
                for seg_ctr in range(nSegments):
                    x_minus_mu_T= np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu= ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr])- .5*(np.dot(x_minus_mu_T,x_minus_mu))

                # Note the max
                logAmax = max(logAjVec.tolist())

                # Calculate the third term from the final eqn in the above Link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)

                # Here Ws are the relative membership weights(p_i/sum(p_i)), but computed in a round-about way
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm)
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)

            # Maximization step
            # print(".join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration:',str(iteration+1), ' M-step']))
            print(f'Image: {imgName} nSegments: {nSegments} iteration: {iteration+1} M-step')

            # temporary reinitialize all pi and mu to 0, before they are recomputed
            mu = np.zeros((nSegments, nColors), dtype='float')
            pi = np.zeros((nSegments,1),dtype='float')

            # Update pi and mu
            for seg_ctr in range(nSegments):
                
                denominatorSum = 0
                # for pix_ctr in range(nPixels):
                #     # Update RGB color vector of mu[seg_ctr] as current mu[seg_ctr] + pixels[pix_ctr,:] times Ws[pix_ctr,seg_ctr]
                #     mu[seg_ctr] = mu[seg_ctr] + np.transpose(pixels[pix_ctr,:]) * Ws[pix_ctr][seg_ctr]

                #     # Update denominatorSum as current denominatorSum + Ws[pix_ctr][seg_ctr]
                #     denominatorSum = denominatorSum + Ws[pix_ctr][seg_ctr]


                """
                Compute mu[seg_ctr] and denominatorSum directly without the for Loop-- 10 points.
                If you find the replacement instruction, comment out the for Loop with your solution
                Hint: Use functions squeeze, tile and reshape along with sum
                """
                mu[seg_ctr] = np.reshape(np.dot(np.squeeze(pixels).T,Ws[:,seg_ctr]), (nColors,))
                denominatorSum = np.sum(Ws[:,seg_ctr])

                # Update mu[seg_ctr] as mu[seg_ctr] divided by denominatorSum
                mu[seg_ctr, : ] = mu[seg_ctr, :] / denominatorSum

                ## Update pi
                # sum of weights (each weight is a probability) for given segment/total num of pixels
                pi[seg_ctr] = denominatorSum / nPixels;

            print(np.transpose(pi))
            print(mu)

            muDiffSq=np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))

            # check for convergence
            if (muDiffSq < .0000001 and piDiffSq < .0000001):#sign of convergence
                print('Convergence Criteria Met at Iteration: ',iteration, '-- Exitingcode')
                break;

            mu_last_iter = mu;
            pi_last_iter = pi;

            segpixels = np.array(pixels)
            cluster= 0
            for pix_ctr in range(nPixels):
                cluster= np.where(Ws[pix_ctr,:]==max(Ws[pix_ctr,:]))
                vec=np.squeeze(np.transpose(mu[cluster,:])) 
                segpixels[pix_ctr,:]=vec.reshape(vec.shape[0],1)

            """
            Save segmented image at each iteration. For displaying consistent image clusters,
            it would be useful to blur/smoothen these segpixels image using a Gaussian filter.  Prior to smoothing,
            convert segpixels to a Grayscale image, and convert the grayscale image into clusters based on pixel intensities
            """
            # reshape segpixels to obtain R,G, B image
            segpixels=np.reshape(segpixels,(img.shape[0],img.shape[1],nColors))

            # convert segpixels to uint8 gray scale image and convert to grayscale-- 5 points
            segpixels = rgb2gray(segpixels)
            segpixels = (segpixels * 255).astype(np.uint8)

            # Use kmeans from sci-kit learn library to cluster pixels in gray scale segpixels image to *nSegments* clusters-- 10 points
            kmeans = KMeans(n_clusters = nSegments).fit(np.reshape(segpixels, (nPixels, 1)))

            # "reshape kmeans.labels_ output of kmeans to have the same size as segpixels -- 5 points
            seglabels = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (segpixels.shape[0], segpixels.shape[1]))

            # Use np.clip, Gaussian smoothing with sigma =2 and label2rgb functions to smoothen the seglabels image, 
            # and output a float RGB image with pixel values between [0--1]-- 10 points
            seglabels = gaussian(np.clip(label2rgb(seglabels), 0, 1), sigma = 2)
            seglabels = np.clip(seglabels, 0, 1)

            # save the segmented output
            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabels)

"""
    Display the 20th iteration (or final output in case of convergence) 
    segimages with nSegments = 2,3,4,5 for the three images-
    this will be a 3 row X 4 column image matrix-- 15 points
"""



# calculate rows from number of images
rows = len(imgNames)

# calculate columns from number of segment counts
cols = len(SegCounts)

# Create a 3-row × 4-column figure
fig, axes = plt.subplots(rows, cols, figsize=(12, 9))

for row, imgName in enumerate(imgNames):  # Iterate over 3 images (rows)
    for col, SegCount in enumerate(SegCounts):  # Iterate over 4 segment counts (columns)
        outputPath = f"./output/{SegCount}_segments/{imgName}/"
        
        # Identify the last iteration's image
        image_files = sorted([f for f in os.listdir(outputPath) if f.endswith(".png")])

        # extract the iteration number from the file name
        iteration_numbers = [int(f.split(".")[0]) for f in image_files]

        # extract the highest iteration number
        max_iteration = max(iteration_numbers)

        if not image_files:
            print(f"Warning: No segmented images found in {outputPath}. Skipping...")
            continue  # Skip to the next image-segment pair
        
        # Load the final image
        final_image_path = join(outputPath, f"{max_iteration}.png")

        # Read and display the image
        img = plt.imread(final_image_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{imgName}, Seg={SegCount}, Iter={max_iteration}")
        axes[row, col].axis("off")  # Hide axis

# Adjust layout and show the figure
plt.tight_layout()
plt.savefig('./output/final.png')
plt.show()





            

                




