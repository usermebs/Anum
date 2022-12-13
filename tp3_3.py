from scipy.integrate import odeint
import numpy as np 
import matplotlib.pyplot as plt 
from math import sqrt

#Q4 : la routine d'approximation des solutions

def approx(m1,m2,k1,k2,w, ci=[0,0,0,0],limit=(0,100,1000),solbreak=False):
    """ retourne l'approximation numérique sur 1000 points par défaut"""

    def F(y,t,a1,a2,b1,b2):
        x1,x2,dx1,dx2=y
        dy= [dx1, dx2 , (1/a1)*(-b1*x1 + b2*((x1-x2))) , (1/a2)*(-b2*x1 - b2*x2 + np.sin(w*t) )] 
        return dy
    a1 , a2 , b1 , b2 = m1 , m2 , k1 , k2

    t=np.linspace(limit[0],limit[1],limit[2])

    sol=odeint(F , ci , t , args=(a1,a2,b1,b2))
    X1=[sol[i][0] for i in range(len(sol))]
    X2=[sol[i][1] for i in range(len(sol))]
    res=[(sol[i][0],sol[i][1]) for i in range(len(sol))]
    if solbreak : return res , X1 , X2
    else : return res

def graphes_de( args* ):
    """ cette fonction permet de tracer les graphes x1(t) 
    et x2(t) simultanement pour differents cas """
    t=np.linspace(0,100,1000)
    plt.figure()
    for i in range(len(args)):   
        plt.plot(t,args[i][1],label='x1_'+str(i)+'(t)')
        plt.plot(t,args[i][2],label='x2_'+str(i)+'(t)')
    plt.legend(loc='best')
    plt.ylabel('position x_1,2(t)')
    plt.xlabel('temps t')
    plt.title('graphes de x1(t) et x2(t)')
    plt.grid()
    plt.show()
    
#calcul l'approximation numérique pour les parametres données
sol1=approx(1,1,1,1,0.5)
sol2=approx(1,1,1,1,1)
sol3=approx(1,1,1,1,1.5)
sol4=approx(1,1,1,1,2)
sol5=approx(1,1,1,1,2.5)
#tracer le graphes de 
graphes_de(sol1,sol2,sol3,sol4,sol5)

def Max1(m1,m2,k1,k2,w,y0=[0,0,0,0],limit=(0,100,300)):
    X1=approx(m1,m2,k1,k2,w,y0, limit, solbreak=True)[1]
    return max(X1)

"""def MaxSearch (m1,m2,k1,k2,precision=1e-4):
    f= lambda w : Max1(m1,m2,k1,k2,w)
    xu,xl = 0,100 # les intervalles adequats
    ea , i = 100 , 1
    r=(sqrt(5)-1)/2
    d=r*(xu-xl)
    x1,x2 = xl+d,xu-d
    f1,f2 = f(x1) , f(x2)

    while ea>precision :
        if f1>f2 : 
            xl=x2
            x2=x1
            f2=f1
            x1=xl+r*(xu-xl)
            f1=f(x1)
        else :
            xu=x1
            x1=x2
            f1=f2
            x2=xu-r*(xu-xl)
            f2=f(x2)
        if f1>f2 :
            xopt=x1
        else :
            xopt=x2

        ea=(1-r)*abs((xu-xl)/xopt)*100
        i+=1
        
        return xopt"""
