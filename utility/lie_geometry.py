import numpy as np

def lie_inner_product(d=2):
    H = np.zeros((d+3,d+3))
    H[:d,:d] = np.identity(d)
    H[-1,-1] = -1
    H[-2,-3]=-1
    H[-3,-2]=-1
    return H

# Conversion TO LIE
def circle_to_lie(C):
    C_ = C[...,:-1]
    R_ = C[...,-1:]
    C_sql2 = np.einsum('...j,...j->...', C_, C_)[...,None]
    return np.concatenate([C_, 0.5*(C_sql2-np.square(R_)),np.ones(R_.shape), R_,],axis=-1)

def point_to_lie(x):
    xtx = np.einsum("...i,...i->...",x,x)[...,None]
    return np.concatenate([x,0.5*xtx, np.ones(xtx.shape),np.zeros(xtx.shape)], axis=-1)

def plane_to_lie(Ab):
    A = Ab[...,:-1]
    b = Ab[...,-1:]
    norm = np.linalg.norm(A,axis=-1,keepdims=True)
    N = A / norm
    c = b / norm
    return np.concatenate([N,c,np.zeros(c.shape), np.ones(c.shape)],axis=-1)

# Conversion FROM LIE
def lie_to_circle(X):
    X_ = X / X[...,-2][...,None]
    return np.concatenate([X_[...,:-3],X_[...,-1:]],axis=-1)
def lie_to_point(X):
    X_ = X / X[:,-2][:,None]
    return X_[:,:-3]

def lie_to_plane(X):
    X_ = X / X[:,-1][:,None]
    nrm = np.linalg.norm(X_[:,:-3],axis=1,keepdims=True)
    X_ /= nrm
    return X_[:,:-2]

def recover_from_lie(X, atol=1-8):
    point_msk = np.abs(X[:,-1]) < atol
    plane_msk = np.abs(X[:,-2]) < atol
    circl_msk = np.logical_not(point_msk)*np.logical_not(plane_msk)
    return lie_to_circle(X[circl_msk]), lie_to_point(X[point_msk]), lie_to_plane(X[plane_msk])

# QUADRIC INTERSECTION AND APOLLONIUS
def line_quadric_intersection(Ks,H,debug=False):

    #_,_,Vts = np.linalg.svd(As@H)
    #Ks = Vts[...,-2:,:]
    Xs = Ks[...,0,:]
    Ys = Ks[...,1,:]

    # solve quadratic equations:
    Ds = Ys - Xs
    a =     np.einsum("...i,ij,...j->...", Ds,H,Ds)
    b = 2 * np.einsum("...i,ij,...j->...", Ds,H,Xs)
    c =     np.einsum("...i,ij,...j->...", Xs,H,Xs)

    delta = np.square(b)-4*a*c

    l1 = (-b-np.sqrt(delta)) / (2*a)
    l2 = (-b+np.sqrt(delta)) / (2*a)
    
    ls_ = np.array([l1,l2]).T
    K_is = np.broadcast_to(np.arange(l1.shape[0])[:,None],ls_.shape)
    
    ls_vinds = (0. <= ls_) * (ls_ <= 1.)
    ls_v  = ls_[ls_vinds]
    T_indices = K_is[ls_vinds]
    
    lc = np.stack([1-ls_v, ls_v],axis=-1)
    solutions = np.einsum("xj,xjk->xk", lc, Ks[T_indices])
    
    if debug:
        print("Solutions on the lie quadric: ")
        
        sol_ips = np.einsum("xi,ij,xj->x", solutions,H,solutions)
        print(np.absolute(sol_ips).max())
        #print(np.einsum("xci,ij,xcj->xc", solutions,H,solutions))

    return solutions, T_indices

def line_quadric_intersection_single(x, y, H):
    # intersect line (1-lambda) x + lambda y with the lie quadric x_^T H x_ = 0

    a =     (y-x).T @ H @ (y-x)
    b = 2*  (y-x).T @ H @ x
    c =         x.T @ H @ x

    delta = b*b-4*a*c
    l1 = (-b-np.sqrt(delta)) / (2*a)
    l2 = (-b+np.sqrt(delta)) / (2*a)

    return l1, l2

def solve_apollonius_single(X, H):
    U,S,Vt = np.linalg.svd(X@H)
    K = Vt[-2:]

    l1,l2 = line_quadric_intersection(K[0],K[1],H)

    x1 = K.T@np.array([1-l1,l1])
    x2 = K.T@np.array([1-l2,l2])

    return x1, x2

def solve_apollonius(As,H,debug=False):

    _,_,Vts = np.linalg.svd(As@H)
    Ks = Vts[...,-2:,:]
    Xs = Ks[...,0,:]
    Ys = Ks[...,1,:]

    # solve quadratic equations:
    Ds = Ys - Xs
    a =     np.einsum("...i,ij,...j->...", Ds,H,Ds)
    b = 2 * np.einsum("...i,ij,...j->...", Ds,H,Xs)
    c =     np.einsum("...i,ij,...j->...", Xs,H,Xs)

    delta = np.square(b)-4*a*c

    l1 = (-b-np.sqrt(delta)) / (2*a)
    l2 = (-b+np.sqrt(delta)) / (2*a)
    ls = np.stack([np.stack([1-l1,l1],axis=-1),
                   np.stack([1-l2,l2],axis=-1)],axis=-1)

    solutions = np.moveaxis(ls,-2,-1)@Ks

    if debug:
        print("Solutions on the lie quadric: ")
        print(np.allclose(np.einsum("xci,ij,xcj->xc", solutions,H,solutions),0.))

        print("Solutions are apollonius solutions?")
        print(np.allclose(np.einsum("xci,ij,xlj->xcl", As,H,solutions),0.))

    return solutions