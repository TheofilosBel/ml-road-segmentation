import os
import numpy as np
import matplotlib.image as mpimg

# Data standardisation
def get_means_stds(img_names, imgs_path):
  '''
    ### Params:
     * img_names: a list of all image names

    ### Retirns:
      * A tuple like: (means, stds)
  '''
  # Read all images in a list
  all_imgs = []
  for img_name in img_names:
    all_imgs.append(mpimg.imread(os.path.join(imgs_path, img_name)))

  # Convert the list of images to a numpy array
  imgs = np.array([np.array(img) for img in all_imgs])

  # Compute the means and standard deviations for each of the 3 RGB channels
  img_means = []
  img_stds = []
  for i in range(imgs.shape[3]):
    img_means.append(np.mean(imgs[:, :, :, i]))
    img_stds.append(np.std(imgs[:, :, :, i]))

  # Return means and stds
  return img_means, img_stds

