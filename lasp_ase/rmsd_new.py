import numpy as np
import math
    
def rmsd(n,coord1,coord2):
    
    xi = np.empty([n])
    yi = np.empty([n])
    
    x_center = np.empty([3])
    y_center = np.empty([3])
    
    x = coord1.copy()
    y = coord2.copy()
    
    Rmatrix = np.empty([3,3])
    
    S = np.empty([4,4])
    dS = np.empty([4,4])
    
    g = np.empty([n,3])
    grd = np.empty([n,3])
    
    tmp = np.empty([3])
    
    q = np.empty([4])
    
    lambda_ = 0
    
    x_norm = 0
    y_norm = 0
    
    for i in range(0,3):
        xi[:] = x[:,i]
        yi[:] = y[:,i]
        x_center[i] = sum(xi[:])/float(n)
        y_center[i] = sum(yi[:])/float(n)
        xi[:] = xi[:] - x_center[i]
        yi[:] = yi[:] - y_center[i]
        x[:,i] = xi[:]
        y[:,i] = yi[:]
        x_norm = x_norm + np.dot(xi,xi)
        y_norm = y_norm + np.dot(yi,yi)
    
    for i in range(0,3):
        for j in range(0,3):
            Rmatrix[i,j] = np.dot(x[:,i],y[:,j])
    
    S[0, 0] = Rmatrix[0, 0] + Rmatrix[1, 1] + Rmatrix[2, 2]
    S[1, 0] = Rmatrix[1, 2] - Rmatrix[2, 1]
    S[2, 0] = Rmatrix[2, 0] - Rmatrix[0, 2]
    S[3, 0] = Rmatrix[0, 1] - Rmatrix[1, 0]
     
    S[0, 1] = S[1, 0]     
    S[1, 1] = Rmatrix[0, 0] - Rmatrix[1, 1] - Rmatrix[2, 2]     
    S[2, 1] = Rmatrix[0, 1] + Rmatrix[1, 0]     
    S[3, 1] = Rmatrix[0, 2] + Rmatrix[2, 0]     
    
    S[0, 2] = S[2, 0]     
    S[1, 2] = S[2, 1]     
    S[2, 2] =-Rmatrix[0, 0] + Rmatrix[1, 1] - Rmatrix[2, 2]     
    S[3, 2] = Rmatrix[1, 2] + Rmatrix[2, 1]     
     
    S[0, 3] = S[3, 0]     
    S[1, 3] = S[3, 1]     
    S[2, 3] = S[3, 2]     
    S[3, 3] =-Rmatrix[0, 0] - Rmatrix[1, 1] + Rmatrix[2, 2] 

    lambda_,q = dstmev(S) ###
    
    U = rotation_matrix(q) ###
    
    error_ = math.sqrt(max(0.0,((x_norm+y_norm)-2.0*lambda_))/float(n))+1E-9
         
    for i in range(0,n):
        for j in range(0,3):
            tmp[:] = np.matmul(np.transpose(U[:,:]),y[i,:])
            g[i,j] = ((x[i,j]) - tmp[j])/(error_*float(n))
            
    for i in range(0,n):
        if abs(math.sqrt(g[i,0]**2+g[i,1]**2+g[i,2]**2) - 0.0000) >= 0.0001:
            for j in range(0,3):
                grd[i,j] = g[i,j]/math.sqrt(g[i,0]**2+g[i,1]**2+g[i,2]**2)
        else:
            for j in range(0,3):
                grd[i,j] = g[i,j]
    return error_, grd, g
    
def rotation_matrix(q):
    
    U = np.empty([3,3])
    
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]     
    q3 = q[3]     
         
    b0 = 2.0*q0     
    b1 = 2.0*q1     
    b2 = 2.0*q2     
    b3 = 2.0*q3     
         
    q00 = b0*q0-1.0   
    q01 = b0*q1     
    q02 = b0*q2     
    q03 = b0*q3     
         
    q11 = b1*q1     
    q12 = b1*q2     
    q13 = b1*q3     
         
    q22 = b2*q2     
    q23 = b2*q3     
         
    q33 = b3*q3     
         
    U[0,0] = q00+q11     
    U[0,1] = q12-q03     
    U[0,2] = q13+q02     
        
    U[1,0] = q12+q03     
    U[1,1] = q00+q22     
    U[1,2] = q23-q01     
    
    U[2,0] = q13-q02     
    U[2,1] = q23+q01     
    U[2,2] = q00+q33     
    
    return U
    
def dstmev(A):
    
    #SV = np.empty([4,4])
    #SW = np.empty(4)
    #rv1 = np.empty(8)
    T,V = givens4(A)   ###
    lambda_=min(T[0,0]-abs(T[0,1]),-abs(T[1,0])+T[1,1]-abs(T[1,2]),
            -abs(T[2,1])+T[2,2]-abs(T[2,3]),-abs(T[3,2])+T[3,3])     
    
    for i in range(0,4):
        T[i,i] = T[i,i] - lambda_
    
    #T,SW,SV,rv1 = svdcmp(4,T,4,4,SW,SV,rv1) 
    T,SW,SV = np.linalg.svd(T)
    
    max_loc = np.argmax(SW)
    lambda_ = SW[max_loc] + lambda_
    
    evec = np.matmul(V,SV[max_loc,:])
    
    return lambda_, evec

def givens4(S):
             
    T = np.empty([4,4])         
    V = np.empty([4,4])         
    T[:,:] = S[:,:]         
    V[:,:] = 0         
    r1 = pythag(T[2,0],T[3,0])         
    if (r1 != 0.) :         
        c1 = T[2,0]/r1; s1 = T[3,0]/r1         
        V[2,2] = c1  ; V[2,3] = s1         
        V[3,2] =-s1  ; V[3,3] = c1          
        T[2,0] = r1  ; T[3,0] = 0. 
        T[2:4,1:4] = np.matmul(V[2:4,2:4],T[2:4,1:4]) 
        T[0:2,2:4] = np.transpose(T[2:4,0:2]) 
        T[2:4,2:4] = np.matmul(T[2:4,2:4],np.transpose(V[2:4,2:4])) 
    else:
        c1 = 1.0
        s1 = 0.0
    
    r2 = pythag(T[2,0],T[1,0])
    if (r2 != 0.):  
       c2 = T[1,0]/r2; s2 = T[2,0]/r2     
       V[1,1] = c2  ; V[1,2] = s2     
       V[2,1] =-s2  ; V[2,2] = c2     
       T[1,0] = r2  ; T[2,0] = 0.0     
       T[1:3,1:4] = np.matmul(V[1:3,1:3],T[1:3,1:4])     
       T[0,1:3]   = T[1:3,0];  T[3,1:3] = T[1:3,3]     
       T[1:3,1:3] = np.matmul(T[1:3,1:3],np.transpose(V[1:3,1:3]))     
    else:     
       c2 = 1.0 
       s2 = 0.0     
    
    r3 = pythag(T[3,1], T[2,1])
    if(r3 != 0.0):   
       c3 = T[2,1]/r3; s3 = T[3,1]/r3     
       V[2,2] = c3  ; V[2,3] = s3     
       V[3,2] =-s3  ; V[3,3] = c3     
       T[2,1] = r3  ; T[3,1] = 0.0     
       T[2:4,2:4] = np.matmul(V[2:4,2:4],T[2:4,2:4])     
       T[0:2,2:4] = np.transpose(T[2:4,0:2])     
       T[2:4,2:4] = np.matmul(T[2:4,2:4],np.transpose(V[2:4,2:4]))     
    else:    
       c3 = 1.0; s3 = 0.0     
    
    V[0,0]=1.0; V[0,1:4] = 0.0; V[1:4,0] = 0.0
    V[1,1] = c2;  V[2,1] = c1*s2 ; V[3,1] = s1*s2; c1c2 = c1*c2; s1c2=s1*c2     
    V[1,2] = -s2*c3 ; V[2,2] = c1c2*c3-s1*s3 ; V[3,2] =  s1c2*c3+c1*s3     
    V[1,3] =  s2*s3 ; V[2,3] =-c1c2*s3-s1*c3 ; V[3,3] = -s1c2*s3+c1*c3     
         
    return T,V    
    
def svdcmp(mmax,a,m,n,w,v,rv1):
    
    g = 0.0
    scale = 0.0
    anorm = 0.0
    for i in range(0,n):
        l=i+1
        rv1[i]=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i <= m):
            for k in range(i,m):
                scale=scale+abs(a[k,i])
            if(scale != 0.0):
                for k in range(i,m):
                    a[k,i]=a[k,i]/scale
                    s=s+a[k,i]*a[k,i]
                f=a[i,i]
                if f >= 0:
                    g = -math.sqrt(s)
                elif f < 0:
                    g = math.sqrt(s)
                h=f*g-s
                a[i,i]=f-g
                for j in range(l,n):
                    s=0.0
                    for k in range(i,m):
                        s=s+a[k,i]*a[k,j]
                    f=s/h
                    for k in range(i,m):
                        a[k,j]=a[k,j]+f*a[k,i]
                for k in range(i,m):
                    a[k,i]=scale*a[k,i]
        w[i]=scale *g
        g=0.0
        s=0.0
        scale=0.0
        if((i<= m) and (i !=n )):
           for k in range(l,n):
              scale=scale+abs(a[i,k])
           if(scale != 0.0):
                for k in range(l,n):
                    a[i,k]=a[i,k]/scale
                    s=s+a[i,k]*a[i,k]
                f=a[i,l]
                if f >= 0:
                    g = -math.sqrt(s)
                elif f < 0:
                    g = math.sqrt(s)
                h=f*g-s
                a[i,l]=f-g
                for k in range(l,n):
                    rv1[k]=a[i,k]/h
                for i in range(l,m):
                    s=0.0
                    for k in range(l,n):
                       s=s+a[j,k]*a[i,k]
                    for k in range(l,n):
                       a[j,k]=a[j,k]+s*rv1[k]
                for i in range(l,n):
                    a[i,k]=scale*a[i,k]
        anorm=max(anorm,(abs(w[i])+abs(rv1[i])))

    for i in range(n-1,-1,-1):
        if(i < n):
            if(g != 0.0):
              for j in range(l,n):
                 v[j,i]=(a[i,j]/a[i,l])/g
              for j in range(l,n):
                 s=0.0
                 for k in range(l,n):
                    s=s+a[i,k]*v[k,j]
                 for k in range(l,n):
                    v[k,j]=v[k,j]+s*v[k,i]
            for j in range(l,n):
               v[i,j]=0.0
               v[j,i]=0.0
        v[i,i]=1.0
        g=rv1[i]
        l=i
    
    for i in range(min(m,n)-1,-1,-1):
        l=i+1
        g=w[i]
        for j in range(l,n):
           a[i,j]=0.0
        if(g != 0.0):
            g=1.0/g
            for j in range(l,n):
                s=0.0
                for k in range(l,m):
                   s=s+a[k,i]*a[k,j]
                f=(s/a[i,i])*g
                for k in range(i,m):
                   a[k,j]=a[k,j]+f*a[k,i]
            for j in range(i,m):
                a[j,i]=a[j,i]*g
        else:
            for j in range(i,m):
                a[j,i]=0.0
        a[i,i]=a[i,i]+1.0
    
    for k in range(n-1,-1,-1):
        for its in range(0,30):
            for l in range(k,-1,-1):
                nm=l-1
                if((abs(rv1[l])+anorm) == anorm):  #goto 2
                    z=w[k]
                    if(l == k):
                        if(z<0.0):
                            w[k]=-z
                            for j in range(0,n):
                                v[j,k]=-v[j,k]
                        continue
                    if(its==30):
                        print('no convergence in svdcmp')
                        exit()
                    x=w[l]
                    nm=k-1
                    y=w[nm]
                    g=rv1[nm]
                    h=rv1[k]
                    f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
                    g=pythag(f,1.0)
                    if f >= 0:
                        calc = abs(g)
                    elif f < 0:
                        calc = -abs(g)
                    f=((x-z)*(x+z)+h*((y/(f+calc))-h))/x 
                    c=1.0
                    s=1.0
                    for j in range(l-1,nm):
                        i=j+1
                        g=rv1[i]
                        y=w[i]
                        h=s*g
                        g=c*g
                        z=pythag(f,h)
                        rv1[j]=z
                        c=f/z
                        s=h/z
                        f= (x*c)+(g*s)
                        g=-(x*s)+(g*c)
                        h=y*s
                        y=y*c
                        for jj in range(0,n):
                            x=v[jj,j]
                            z=v[jj,i]
                            v[jj,j]= (x*c)+(z*s)
                            v[jj,i]=-(x*s)+(z*c)
                        z=pythag(f,h)
                        w[j]=z
                        if(z!=0.0):
                            z=1.0/z
                            c=f*z
                            s=h*z
                        f= (c*g)+(s*y)
                        x=-(s*g)+(c*y)
                        for jj in range(0,m):
                            y=a[jj,j]
                            z=a[jj,i]
                            a[jj,j]= (y*c)+(z*s)
                            a[jj,i]=-(y*s)+(z*c)
                    rv1[l]=0.0
                    rv1[k]=f
                    w[k]=x
                
                if ((abs(w[nm])+anorm) == anorm):  #goto 1
                    c=0.0
                    s=1.0
                    for i in range(l-1,k):
                        f=s*rv1[i]
                        rv1[i]=c*rv1[i]
                        if((abs(f)+anorm) == anorm): #goto 2
                            z=w[k]
                            if(l == k):
                                if(z<0.0):
                                    w[k]=-z
                                    for j in range(0,n):
                                        v[j,k]=-v[j,k]
                                continue
                            if(its==30):
                                print('no convergence in svdcmp')
                                exit()
                            x=w[l]
                            nm=k-1
                            y=w[nm]
                            g=rv1[nm]
                            h=rv1[k]
                            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
                            g=pythag(f,1.0)
                            if f >= 0:
                                calc = abs(g)
                            elif f < 0:
                                calc = -abs(g)
                            f=((x-z)*(x+z)+h*((y/(f+calc))-h))/x 
                            c=1.0
                            s=1.0
                            for j in range(l-1,nm):
                                i=j+1
                                g=rv1[i]
                                y=w[i]
                                h=s*g
                                g=c*g
                                z=pythag(f,h)
                                rv1[j]=z
                                c=f/z
                                s=h/z
                                f= (x*c)+(g*s)
                                g=-(x*s)+(g*c)
                                h=y*s
                                y=y*c
                                for jj in range(0,n):
                                    x=v[jj,j]
                                    z=v[jj,i]
                                    v[jj,j]= (x*c)+(z*s)
                                    v[jj,i]=-(x*s)+(z*c)
                                z=pythag(f,h)
                                w[j]=z
                                if(z!=0.0):
                                    z=1.0/z
                                    c=f*z
                                    s=h*z
                                f= (c*g)+(s*y)
                                x=-(s*g)+(c*y)
                                for jj in range(0,m):
                                    y=a[jj,j]
                                    z=a[jj,i]
                                    a[jj,j]= (y*c)+(z*s)
                                    a[jj,i]=-(y*s)+(z*c)
                            rv1[l]=0.0
                            rv1[k]=f
                            w[k]=x
                        
                        g=w[i]
                        h=pythag(f,g)
                        w[i]=h
                        h=1.0/h
                        c= (g*h)
                        s=-(f*h)
                        for j in range(0,m):
                            y=a[j,nm]
                            z=a[j,i]
                            a[j,nm]=(y*c)+(z*s)
                            a[j,i]=-(y*s)+(z*c)
                    z=w[k]
                    if(l == k):
                        if(z<0.0):
                            w[k]=-z
                            for j in range(0,n):
                                v[j,k]=-v[j,k]
                        continue
                    if(its==30):
                        print('no convergence in svdcmp')
                        exit()
                    x=w[l]
                    nm=k-1
                    y=w[nm]
                    g=rv1[nm]
                    h=rv1[k]
                    f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
                    g=pythag(f,1.0)
                    if f >= 0:
                        calc = abs(g)
                    elif f < 0:
                        calc = -abs(g)
                    f=((x-z)*(x+z)+h*((y/(f+calc))-h))/x 
                    c=1.0
                    s=1.0
                    for j in range(l-1,nm):
                        i=j+1
                        g=rv1[i]
                        y=w[i]
                        h=s*g
                        g=c*g
                        z=pythag(f,h)
                        rv1[j]=z
                        c=f/z
                        s=h/z
                        f= (x*c)+(g*s)
                        g=-(x*s)+(g*c)
                        h=y*s
                        y=y*c
                        for jj in range(0,n):
                            x=v[jj,j]
                            z=v[jj,i]
                            v[jj,j]= (x*c)+(z*s)
                            v[jj,i]=-(x*s)+(z*c)
                        z=pythag(f,h)
                        w[j]=z
                        if(z!=0.0):
                            z=1.0/z
                            c=f*z
                            s=h*z
                        f= (c*g)+(s*y)
                        x=-(s*g)+(c*y)
                        for jj in range(0,m):
                            y=a[jj,j]
                            z=a[jj,i]
                            a[jj,j]= (y*c)+(z*s)
                            a[jj,i]=-(y*s)+(z*c)
                    rv1[l]=0.0
                    rv1[k]=f
                    w[k]=x
            
            c=0.0
            s=1.0    
            for i in range(l-1,k):    
                f=s*rv1[i]    
                rv1[i]=c*rv1[i]    
                if((abs(f)+anorm) == anorm): #goto 2    
                    z=w[k]    
                    if(l == k):    
                        if(z<0.0):    
                            w[k]=-z    
                            for j in range(0,n):    
                                v[j,k]=-v[j,k]    
                        continue    
                    if(its==30):    
                        print('no convergence in svdcmp')    
                        exit()    
                    x=w[l]    
                    nm=k-1    
                    y=w[nm]    
                    g=rv1[nm]    
                    h=rv1[k]    
                    f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)    
                    g=pythag(f,1.0)    
                    if f >= 0:    
                        calc = abs(g)    
                    elif f < 0:    
                        calc = -abs(g)    
                    f=((x-z)*(x+z)+h*((y/(f+calc))-h))/x     
                    c=1.0    
                    s=1.0    
                    for j in range(l-1,nm):    
                        i=j+1    
                        g=rv1[i]    
                        y=w[i]    
                        h=s*g    
                        g=c*g    
                        z=pythag(f,h)    
                        rv1[j]=z    
                        c=f/z    
                        s=h/z    
                        f= (x*c)+(g*s)    
                        g=-(x*s)+(g*c)    
                        h=y*s    
                        y=y*c    
                        for jj in range(0,n):    
                            x=v[jj,j]    
                            z=v[jj,i]    
                            v[jj,j]= (x*c)+(z*s)    
                            v[jj,i]=-(x*s)+(z*c)    
                        z=pythag(f,h)    
                        w[j]=z    
                        if(z!=0.0):    
                            z=1.0/z    
                            c=f*z    
                            s=h*z    
                        f= (c*g)+(s*y)    
                        x=-(s*g)+(c*y)    
                        for jj in range(0,m):    
                            y=a[jj,j]    
                            z=a[jj,i]    
                            a[jj,j]= (y*c)+(z*s)    
                            a[jj,i]=-(y*s)+(z*c)    
                    rv1[l]=0.0    
                    rv1[k]=f    
                    w[k]=x    
                    
                g=w[i]    
                h=pythag(f,g)    
                w[i]=h    
                h=1.0/h    
                c= (g*h)    
                s=-(f*h)    
                for j in range(0,m):    
                    y=a[j,nm]    
                    z=a[j,i]    
                    a[j,nm]=(y*c)+(z*s)    
                    a[j,i]=-(y*s)+(z*c)    
            z=w[k]    
            if(l == k):    
                if(z<0.0):    
                    w[k]=-z    
                    for j in range(0,n):    
                        v[j,k]=-v[j,k]    
                continue    
            if(its==30):    
                print('no convergence in svdcmp')    
                exit()    
            x=w[l]    
            nm=k-1    
            y=w[nm]    
            g=rv1[nm]    
            h=rv1[k]    
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)    
            g=pythag(f,1.0)    
            if f >= 0:
                calc = abs(g)
            elif f < 0:
                calc = -abs(g)
            f=((x-z)*(x+z)+h*((y/(f+calc))-h))/x 
            c=1.0
            s=1.0
            for j in range(l-1,nm):
                i=j+1
                g=rv1[i]
                y=w[i]
                h=s*g
                g=c*g
                z=pythag(f,h)
                rv1[j]=z
                c=f/z
                s=h/z
                f= (x*c)+(g*s)
                g=-(x*s)+(g*c)
                h=y*s
                y=y*c
                for jj in range(0,n):
                    x=v[jj,j]
                    z=v[jj,i]
                    v[jj,j]= (x*c)+(z*s)
                    v[jj,i]=-(x*s)+(z*c)
                z=pythag(f,h)
                w[j]=z
                if(z!=0.0):
                    z=1.0/z
                    c=f*z
                    s=h*z
                f= (c*g)+(s*y)
                x=-(s*g)+(c*y)
                for jj in range(0,m):
                    y=a[jj,j]
                    z=a[jj,i]
                    a[jj,j]= (y*c)+(z*s)
                    a[jj,i]=-(y*s)+(z*c)
            rv1[l]=0.0
            rv1[k]=f
            w[k]=x
        continue
    return a,w,v,rv1

def pythag(a,b):    
                   
    absa=abs(a)    
    absb=abs(b)    
    if(absa > absb):
       pythag=absa*math.sqrt(1.0+(absb/absa)**2)
    else:       
        if(absb==0.0):   
            pythag=0.0
        else:   
            pythag=absb*math.sqrt(1.0+(absa/absb)**2)
    
    return pythag








         
