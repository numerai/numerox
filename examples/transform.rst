Transform features
==================

Numerox offers several ways to transform features (x).

PCA
---

You can use principal component analysis (PCA) to make the features
orthogonal::

    >>> data2 = data.pca()

You can only keep the number of orthogonal features that explain at least,
say, 90% of the variance::

    >>> data2 = data.pca(nfactor=0.9)

which for the dataset I am using leaves me with 15 features. I'd get the
same result if I had used ``nfactor=15``.

You can fit the PCA on, say, the train data and then use that fit to transform
all the data::

    >>> data2 = data.pca(nfactor=0.9, data_fit=data['train'])
