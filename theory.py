import numpy as np
import scipy as sp
from scipy.special import erf as erf

def relu(x):
    return x * (x > 0)

def VReLU(cov, v, v2=None):
    '''
    Computes E[ReLU(z1) ReLU(z2)]
    where (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2 is None:
        c = cov/v
        outv = v
    else:
        outv = np.sqrt(v * v2)
        c = cov/outv
    return (0.5 / np.pi) * (np.sqrt(1 - c**2) + (np.pi - np.arccos(c)) * c) * outv
def VStep(cov, v, v2=None):
    '''
    Computes E[Step(z1) Step(z2)]
    where Step is the function takes positive numbers to 1 and
    all else to 0, and 
    (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2 is None:
        c = cov/v
    else:
        c = cov/np.sqrt(v * v2)
    return (0.5 / np.pi) * (np.pi - np.arccos(c))
def VErf(cov, v, v2=None):
    '''
    Computes E[erf(z1)erf(z2)]
    where (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2:
        return 2/np.pi * np.arcsin(cov/np.sqrt((v+0.5)(v2+0.5)))
    else:
        return 2/np.pi * np.arcsin(cov/(v+0.5))
def VDerErf(cov, v, v2=None):
    '''
    Computes E[erf'(z1)erf'(z2)]
    where (z1, z2) is sampled from a 2-dimensional Gaussian
    with mean 0 and covariance
    |v    cov|
    |cov  v  |
    or
    |v    cov|
    |cov  v2 |
    
    Inputs:
        `cov`: covariance of input matrix
        `v`: common diagonal variance of input matrix, if `v2` is None;
            otherwise, the first diagonal variance
        `v2`: the second diagonal variance
    The inputs can be tensors, in which case they need to have the same shape
    '''
    if v2:
        return 4/np.pi * ((1+2*v)*(1+2*v2) - 4 * cov**2)**-0.5
    else:
        return 4/np.pi * ((1+2*v)**2 - 4 * cov**2)**-0.5
    
def boolcubeFgrid(dim, layer, vw, vb, Vfun, Vderfun):
    '''
    If we have an multilayer perceptron with activation function "fun"
    and initialized by
    $$W_{ij}^l \sim N(0, `vw`/width), b_i^l \sim N(0, `vb`)$$
    then this function computes the unique values that
    the CK and the NTK kernel functions take on the boolean cube with
    `dim` dimensions, over all depth up to `layer`.
    Note importantly that we assume the input space is the standard
    boolean cube {1, -1}^d (and not a scaled version of it).
    
    Inputs:
        `dim`: integer, dimension of boolean cube
        `layer`: the number of layers to consider; we return all kernel values
            up to this layer
        `vw`: variance of weights. It should be a tensor
        `vb`: variance of biases. It should be a tensor of the same shape as `vw`
        `Vfun`: the V-transform of the activation function.
            Should take 2 tensors of equal shapes as inputs
            (see `VErf` for example)
        `Vderfun`: the V-transform of the derivative of the activation function.
            Should take 2 tensors of equal shapes as inputs
            (see `VDerErf` for example)
    Outputs:
        a dictionary
        ```{"cks": cks, "ntks": ntks}```
        where
        `cks` has shape [1] + `vw.shape` + [1], 
            gives the values of the CK on the boolean cube in the last dimension,
            and varies by layer in the first dimension;
        `ntks` has shape [1] + `vw.shape` + [1], 
            gives the values of the NTK on the boolean cube in the last dimension,
            and varies by layer in the first dimension.
    '''
    # compute diagonal variances
    diags = []
    # assuming normalized input variance = 1
    diags.append(vw + vb)
    for i in range(layer):
        diags.append(vw * Vfun(diags[-1], diags[-1]) + vb)
    # diags[l] is depth-l network (normalized) variance
    
    vw_ = vw[..., None]
    vb_ = vb[..., None]
    
    # Compute ck
    covs = []
    basegrid = np.broadcast_to(
            np.linspace(-1, 1, dim+1), tuple([1]*vw.ndim+[dim+1]))
    covs.append(
        vw_ * basegrid + vb_)
    for i in range(layer):
        covs.append(vw_ * Vfun(covs[-1], diags[i][..., None]) + vb_)
    
    # Computing NTK
    ntks = []
    ntks.append(covs[0])
    for i in range(layer):
        ntks.append(
            covs[i+1] +
            vw_ * ntks[-1] * Vderfun(covs[i], diags[i][..., None])
        )
    return {"cks": np.array(covs), "ntks": np.array(ntks)}

def boolCubeMu(dim, deg, covs,
               normalize=True, fastAdd=True, fastDiff=False, twostep=True):
    '''
    For a neural kernel whose unique values over the `dim`-dimensional
    boolean cube are given in `covs`, return the eigenvalue associated to
    degree `deg`.
    
    Inputs:
        `dim`: dimension of boolean cube
        `deg`: degree of the eigenvalue
            (recall that each unique eigenvalue over the uniform distribution of
            the boolean cube is associated to a particular degree
            from 0 to `dim`)
        `covs`: a tensor whose last dimension gives the values of the NN kernel
            on {-1, -1 + 2/dim, -1 + 4/dim..., 1 - 2/dim, 1}
        `normalize`: whether to give normalized eigenvalues, such that
            the sum of eigenvalues (with multiplicity) equals 1.
            Default: True
        Algorithmic parameters:
        `fastAdd`: whether to perform finite addition using a binomial formula.
            Default: True
        `fastDiff`: whether to perform finite difference using a binomial formula.
            Note that this can cause numerical instability.
            Default: False
        `twostep`: whether to combine a finite addition and a finite difference
            into a single finite difference step with twice the step size.
            Default: True
    Returns:
        an array with the same dimension as covs[..., 0], containing the
        corresponding eigenvalues.
    '''
    FDs = covs
    if twostep:
        if 2*deg <= dim:
            if fastDiff:
                binomialcoefs = np.array(
                    [sp.special.binom(deg, k) / 2**deg * (-1)**(deg-k)
                     for k in range(deg+1)])
                binommatrix = np.zeros([dim+1, dim-2*deg+1])
                for i in range(dim-2*deg+1):
                    binommatrix[i:i+2*deg+1:2, i] = binomialcoefs
                FDs = covs @ binommatrix
            else:
                # compute the finite differences with step 2*delta
                for _ in range(deg):
                    FDs = (FDs[..., 2:] - FDs[..., :-2])/4
            # FDs is now `deg`th finite difference of `covs` with step size 2*delta
            if fastAdd:
                binomialcoefs = np.array(
                    [sp.special.binom(dim-2*deg, k) / 2**(dim-2*deg)
                     for k in range(dim-2*deg+1)])
                if normalize:
                    return FDs @ binomialcoefs / covs[..., -1]
                else:
                    return FDs @ binomialcoefs
            else:
                # doing addition recursively to preserve numeric stability
                for _ in range(dim - 2 * deg):
                    FDs = (FDs[..., :-1] + FDs[..., 1:])/2
                if normalize:
                    return FDs[..., 0] / covs[..., -1]
                else:
                    return FDs[..., 0]
        else:
            if fastDiff:
                raise NotImplemented()
            # compute the finite differences with step 2*delta
            for _ in range(dim - deg):
                FDs = (FDs[..., 2:] - FDs[..., :-2])/4
            for _ in range(2*deg - dim):
                FDs = (FDs[..., 1:] - FDs[..., :-1])/2
            if normalize:
                return FDs[..., 0] / covs[..., -1]
            else:
                return FDs[..., 0]            
    else:
        if fastDiff:
            binomialcoefs = np.array(
                [sp.special.binom(deg, k) / 2**deg * (-1)**(deg-k)
                 for k in range(deg+1)])
            binommatrix = np.zeros([dim+1, dim-deg+1])
            for i in range(dim-deg+1):
                binommatrix[i:i+deg+1, i] = binomialcoefs
            FDs = covs @ binommatrix
        else:
            # compute the finite differences with step delta
            for _ in range(deg):
                FDs = (FDs[..., 1:] - FDs[..., :-1])/2
        # FDs is now `deg`th finite difference of `covs`
        if fastAdd:
            binomialcoefs = np.array(
                [sp.special.binom(dim-deg, k) / 2**(dim-deg)
                 for k in range(dim-deg+1)])
            if normalize:
                return FDs @ binomialcoefs / covs[..., -1]
            else:
                return FDs @ binomialcoefs
        else:
            # doing addition recursively to preserve numeric stability
            for _ in range(dim - deg):
                FDs = (FDs[..., :-1] + FDs[..., 1:])/2
            if normalize:
                return FDs[..., 0] / covs[..., -1]
            else:
                return FDs[..., 0]
            
def boolCubeMuAll(dim, deg, covs,
               normalize=True, twostep=True):
    '''
    For a neural kernel whose unique values over the `dim`-dimensional
    boolean cube are given in `covs`, return all eigenvalue associated to
    degree less than or equal to `deg`.
    This function reuses finite difference computation so that it is
    much more efficient compared to computing the individual eigenvalues
    each with `boolCubeMu`.
    
    Inputs:
        `dim`: dimension of boolean cube
        `deg`: maximum degree of the eigenvalue
            (recall that each unique eigenvalue over the uniform distribution of
            the boolean cube is associated to a particular degree
            from 0 to `dim`)
        `covs`: a tensor whose last dimension gives the values of the NN kernel
            on {-1, -1 + 2/dim, -1 + 4/dim..., 1 - 2/dim, 1}
        `normalize`: whether to give normalized eigenvalues, such that
            the sum of eigenvalues (with multiplicity) equals 1.
            Default: True
        Algorithmic parameters:
        `twostep`: whether to combine a finite addition and a finite difference
            into a single finite difference step with twice the step size.
            Note that this currently requires 2*`deg` <= `dim`.
            Default: True
    Returns:
        an array with the same dimension as covs[..., 0], containing the
        corresponding eigenvalues.
    '''
    eigens = []
    FDs = [covs]
    if twostep:
        if 2 * deg > dim:
            raise NotImplementedError(
                "`twostep` algorithm does not currently work for the case when 2 * deg > dim")
        # compute the finite differences with step 2*delta
        for _ in range(deg):
            FDs.append((FDs[-1][..., 2:] - FDs[-1][..., :-2])/4)
        for d in range(deg+1):
            binomialcoefs = np.array(
                [sp.special.binom(dim-2*d, k) / 2**(dim-2*d)
                 for k in range(dim-2*d+1)])
            if normalize:
                eigens.append(FDs[d] @ binomialcoefs / covs[..., -1])
            else:
                eigens.append(FDs[d] @ binomialcoefs)
    else:
        for _ in range(deg):
            FDs.append((FDs[-1][..., 1:] - FDs[-1][..., :-1])/2)
        for d in range(deg+1):
            binomialcoefs = np.array(
                [sp.special.binom(dim-d, k) / 2**(dim-d)
                 for k in range(dim-d+1)])
            if normalize:
                eigens.append(FDs[d] @ binomialcoefs / covs[..., -1])
            else:
                eigens.append(FDs[d] @ binomialcoefs)
    return np.array(eigens)
        