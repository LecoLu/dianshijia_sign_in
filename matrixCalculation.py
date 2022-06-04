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
    am,_ =A.shape
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

# from: https://blog.csdn.net/u012958850/article/details/125284113   
def P1(A, i, j, row=True):            
    if row:
        A[[i,j]]=A[[j,i]]              
    else:
        A[:,[i,j]]=A[:,[j,i]]          
def P2(A,i,k, row=True):				
    if row:
        A[i]=k*A[i]						
    else:
        A[:,i]=k*A[:,i]					
def P3(A,i,j,k,row=True):               
    if row:
        A[j]+=k*A[i]                    
    else:
        A[:,j]+=k*A[:,i]                
def rowLadder(A, m, n):
    rank=0                                     
    zero=m                                     
    i=0                                         
    order=np.array(range(n))                  
    while i<min(m,n) and i<zero:                
        flag=False                             
        index=np.where(abs(A[i:,i])>1e-10)     
        if len(index[0])>0:                     
            rank+=1                            
            flag=True                         
            k=index[0][0]                      
            if k>0:                             
                P1(A,i,i+k)                     
        else:                                   
            index=np.where(abs(A[i,i:n])>1e-10)
            if len(index[0])>0:               
                rank+=1
                flag=True
                k=index[0][0]
                P1(A,i,i+k,row=False)           
                order[[i, k+i]]=order[[k+i, i]] 
        if flag:                               
            P2(A,i,1/A[i,i])
            for t in range(i+1, zero):
                P3(A,i,t,-A[t,i])
            i+=1                               
        else:                                  
            P1(A,i,zero-1)
            zero-=1                           
    return rank, order         
def simplestLadder(A,rank):
    for i in range(rank-1,0,-1):                
        for j in range(i-1, -1,-1):            
            P3(A,i,j,-A[j,i])
             
def mySolve(A,b):
    m,n=A.shape                                     
    b=b.reshape(b.size, 1)                         
    B=np.hstack((A, b))                           
    r, order=rowLadder(B, m, n)                    
    X=np.array([])                                
    index=np.where(abs(B[:,n])>1e-10)              
    nonhomo=index[0].size>0                        
    r1=r                                           
    if nonhomo:                                    
        r1=np.max(index)+1                        
    solvable=(r>=r1)                               
    if solvable:                                   
        simplestLadder(B, r)                       
        X=np.vstack((B[:r,n].reshape(r,1),         
                            np.zeros((n-r,1))))
        if r<n:                                   
            x1=np.vstack((-B[:r,r:n],np.eye(n-r)))
            X=np.hstack((X,x1))
        X=X[order]
    return X

# dimension of null space
def null_space(A):
    am,an=A.shape
    B=np.zeros(am)
    X=mySolve(A,B)
            
    return np.linalg.matrix_rank(X)

# solve matrix and output two answers
def solve_matrix(A_b):
    return (0,0)
