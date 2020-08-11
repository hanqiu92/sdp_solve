import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix,csc_matrix
import scipy.sparse.linalg as splinalg
import smcp
from sksparse.cholmod import cholesky,Factor
from numba import njit

TOL = 1e-6
ZERO = 1e-16

def process_inputs(fname,model_name):
    ## read inputs
    A,b,_ = smcp.misc.sdpa_read(fname)
    n,m = np.sqrt(A.size[0]).astype(np.int64),b.size[0]

    ## process b
    b_dense = np.array(b)[:,0]
    
    ACp,ACj,ACx = A.CCS
    ACp,ACj,ACx = np.array(ACp)[:,0],np.array(ACj)[:,0],np.array(ACx)[:,0]
    ACrow,ACcol = ACj // n, ACj % n
    A_idx_start = ACp[1]

    ## process C
    Cj,Cx = ACj[:A_idx_start],ACx[:A_idx_start]
    Crow,Ccol = ACrow[:A_idx_start],ACcol[:A_idx_start]
    bool_off_diag = Crow != Ccol
    Cx_off,Crow_off,Ccol_off = Cx[bool_off_diag],Crow[bool_off_diag],Ccol[bool_off_diag]
    Cx = np.concatenate([Cx,Cx_off])
    Crow = np.concatenate([Crow,Ccol_off])
    Ccol = np.concatenate([Ccol,Crow_off])
    C_csc = csc_matrix((Cx,(Crow,Ccol)),shape=(n,n))
    
    ## process A
    Ap,Aj,Ax = ACp[1:] - A_idx_start,ACj[A_idx_start:],ACx[A_idx_start:]
    Arow,Acol = ACrow[A_idx_start:],ACcol[A_idx_start:]
    
    Asp,Asj,Asx = np.zeros((m+1,),dtype=np.int64),np.empty((2*len(Ax),),dtype=np.int64),np.empty((2*len(Ax),),dtype=np.float64)
    nnz = 0
    for k in range(m):
        ranges = slice(Ap[k],Ap[k+1])
        r,c,x = Arow[ranges],Acol[ranges],Ax[ranges]
        bool_off_diag = r != c
        r_off,c_off,x_off = r[bool_off_diag],c[bool_off_diag],x[bool_off_diag]
        r = np.concatenate([r,c_off])
        c = np.concatenate([c,r_off])
        x = np.concatenate([x,x_off])

        nnz += len(x)
        Asp[k+1] = nnz
        Asj[Asp[k]:Asp[k+1]] = r * n + c
        Asx[Asp[k]:Asp[k+1]] = x

    Asj,Asx = Asj[:Asp[-1]],Asx[:Asp[-1]]
    As_csc = csc_matrix((Asx,Asj,Asp),shape=(n*n,m))
    As_h = As_csc.reshape((n,n*m),order='F')
    As_csr = As_csc.T
    As_csc = As_csc.tocsr()

    print('{}: model finish reading. n={}, m={}, nnz={}.'.format(model_name,n,m,len(Ax)))
    return (n,m,C_csc,b_dense,As_h,As_csr,As_csc)

## initial solution
def initialization(As_csc,C_csc,b_dense,n,m):
    ## As_csc is of shape (n*n) * m; C_csc is of shape n * n
    As_norm = np.sqrt(As_csc.multiply(As_csc).sum(axis=0))
    scale_p = max(10,math.sqrt(n),math.sqrt(n)*np.max((1+np.abs(b_dense))/(1+As_norm)))
    scale_d = max(10,math.sqrt(n),f_norm(C_csc),np.max(As_norm))
    X = sp.eye(n,n,dtype=np.float64).tocsc() * scale_p
    S = sp.eye(n,n,dtype=np.float64).tocsc() * scale_d
    AX = X.reshape((n*n,1)).T.dot(As_csc).A[0,:]
    
    rescale_p = min(1,10 * l_norm(b_dense)/l_norm(AX))
    rescale_p = max(rescale_p,10/scale_p)
    X = X * rescale_p
    rescale_d = min(1,10 * f_norm(C_csc)/f_norm(S))
    rescale_d = max(rescale_d,10/scale_d)
    S = S * rescale_d
    print('initialization scale: p={:.2f}, d={:.2f}.'.format(scale_p * rescale_p,scale_d * rescale_d))
    return X,S

def l_norm(v):
    return np.sqrt(np.sum(np.square(v)))

def f_norm(A):
    return l_norm(A.data)

### line search

def check_psd(X):
    try:
        X_inv_op = cholesky(X)
        return (np.min(X_inv_op.D()) >= ZERO)
    except Exception as e:
        return False

def line_search(X,dX):
    ## first check the full step size
    alpha = 1
    is_psd = check_psd(X + dX)
    if is_psd:
        return alpha

    ## next check the eigen value
    try:
        factor = cholesky(X)
        PdXPt = factor.apply_P(factor.apply_P(dX).T)
        LiPdXPtLit = factor.solve_L(factor.solve_L(PdXPt,use_LDLt_decomposition=False).T,use_LDLt_decomposition=False)
        eigs = splinalg.eigsh(LiPdXPtLit,k=5,which='LM',return_eigenvectors=False,maxiter=30)
        alpha = -1 / np.min(eigs)
        if 0 <= alpha < 1:
            alpha = alpha * 0.95
            ## check the validity of the found step size
            is_psd = check_psd(X + alpha * dX)
            if is_psd:
                return alpha        
    except Exception as e:
        pass

    if alpha < 0:
        alpha = 1

    ## otherwise, has to use backtracking to find a valid step size
    backtrack_rate = 0.8
    count = 0
    while not is_psd and count < 50:
        count += 1
        alpha *= backtrack_rate
        is_psd = check_psd(X + alpha * dX)
    if not is_psd:
        return -1

    ## do bisect search to improve the step size
    alpha_lb,alpha_ub = alpha,alpha / backtrack_rate
    alpha = (alpha_lb + alpha_ub) / 2
    for _ in range(10):
        is_psd = check_psd(X + alpha * dX)
        if is_psd:
            alpha_lb = alpha
        else:
            alpha_ub = alpha
        alpha = (alpha_lb + alpha_ub) / 2
    alpha = alpha_lb

    ## finally, keep the solution from boundary
    alpha = alpha * 0.95
    return alpha