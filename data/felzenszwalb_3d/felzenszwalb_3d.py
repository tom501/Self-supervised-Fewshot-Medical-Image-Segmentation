import numpy as np

from felzenszwalb_3d_cy import felzenszwalb_cython_3d


def felzenszwalb_3d(image, scale=1, sigma=0.8, min_size=20, multichannel=True, spacing=(1,1,1)):
    """Computes Felsenszwalb's efficient graph based image segmentation.

    Produces an oversegmentation of a multichannel (i.e. RGB) image
    using a fast, minimum spanning tree based clustering on the image grid.
    The parameter ``scale`` sets an observation level. Higher scale means
    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,
    used for smoothing the image prior to segmentation.

    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    For RGB images, the algorithm uses the euclidean distance between pixels in
    color space.

    Parameters
    ----------
    image : (width, height, 3) or (width, height) ndarray
        Input image.
    scale : float
        Free parameter. Higher means larger clusters.
    sigma : float
        Width (standard deviation) of Gaussian kernel used in preprocessing.
    min_size : int
        Minimum component size. Enforced using postprocessing.
    multichannel : bool, optional (default: True)
        Whether the last axis of the image is to be interpreted as multiple
        channels. A value of False, for a 3D image, is not currently supported.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    References
    ----------
    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004

    Notes
    -----
        The `k` parameter used in the original paper renamed to `scale` here.

    Examples
    --------
    >>> from skimage.segmentation import felzenszwalb
    >>> from skimage.data import coffee
    >>> img = coffee()
    >>> segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)
    """

    # if not multichannel and image.ndim > 2:
    #     raise ValueError("This algorithm works only on single or "
    #                      "multi-channel 2d images. ")

    image = np.atleast_3d(image)
    return felzenszwalb_cython_3d(image, scale=scale, sigma=sigma, min_size=min_size, spacing=spacing)

#
# test = felzenszwalb_3d(img, min_size=5000, sigma=1)
#
# import matplotlib.pyplot as plt
# from skimage.segmentation import slic, watershed, felzenszwalb
#
# image = img
#
# vals = np.linspace(0, 1, np.max((test.max(), test.max())))
# np.random.shuffle(vals)
# cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
#
# for s in range(image.shape[0]):
#     plt.figure(figsize=(15, 10))
#     plt.subplot(1, 4, 1)
#     plt.imshow(image[s], cmap='bone')
#     plt.subplot(1, 4, 2)
#     plt.imshow(lbl[s], vmin=lbl.min(), vmax=lbl.max())
#     plt.subplot(1, 4, 3)
#     plt.imshow(test[s], vmin=test.min(), vmax=test.max(), cmap=cmap)
#     plt.title('supervoxels')
#     plt.subplot(1, 4, 4)
#     plt.imshow(felzenszwalb(image[s], min_size=400, sigma=1), cmap=cmap)
#     plt.title('superpixels')
