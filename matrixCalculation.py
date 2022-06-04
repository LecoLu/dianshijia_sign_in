import numpy as np
import sympy as sp

# add two matrix
def add(A,B):
    return A+B

# subtract two matrix
def sub(A,B):
    return A-B

# def multiply
def multiply(A,B):
    return A@B

# calculate the norm
def norm(A):
    return 0

# calculate the basis of the vector
def basis(A):
    R_a,_ = sp.Matrix(A).rref()
    R = np.array(R_a)
    rm,rn=R.shape
    am,an =A.shape
    X=np.zeros(am)
    rank = 0
    for i in range(rm):
        for j in range(rn):
            if (R[i][j]!=0):
                X[i]=1
                rank +=1
                break
    T=[]
    if(rank==0):
        return []
    for i in range(len(X)):
        if X[i]==1:
            T.append(A[:,i])
    return np.array(T).T

# solve matrix and output two answers
def solve_matrix(A_b):
    return (0,0)