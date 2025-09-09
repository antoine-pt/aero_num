"""
AUTEURS :
    - Antoine Payette-Toupin 2090788
    - Célestin Lesire 2486215
    - Corentin Léon 2484549


Projet LAP1 - Conduction thermique 1D régime permanent
-------------------------------------------------
Ce programme résout plusieurs cas de conduction thermique en régime permanent
dans en 1D et en utilisant la méthode des volumes finis telle que décrite par 
Versteeg et Malalasekera au Chapitre 4 du livre "An Introduction
to Computational Fluid Dynamics".
Trois scénarios sont proposés :
1. Conduction thermique 1D sans source de chaleur.
2. Conduction thermique 1D avec génération uniforme de chaleur.
3. Refroidissement d’une ailette cylindrique par convection.

"""


import numpy as np
import matplotlib.pyplot as plt

## LAP1 Aérodynamique Numérique

## Volume finis

class Data:
    """
    Cette classe regroupe les données physiques du problème de conduction tehrmique 1D :
    - Longueur de la barre
    - Nombre de volumes de contrôle (résolution)
    - Aire de la section
    - Conductivité thermique
    - Températures imposées aux extrémités
    - Génération de chaleur éventuelle (q)
    
    Elle fournit aussi une propriété dx = L/N qui représente la taille d’un élément de maillage.
    """
    def __init__(self,
                 conduction=None,
                 convection=None,
                 velocity=None,
                 rho=None,
                 heatGeneration=None,
                 phiW=None,
                 phiE=None,
                 Tinf=None,
                 qW=None,
                 qE=None,
                 length=None,
                 area=None,
                 resolution=None):
        self.conduction = conduction
        self.convection = convection
        self.heatGeneration = heatGeneration
        self.velocity = velocity
        self.rho = rho
        self.phiW = phiW
        self.phiE = phiE
        self.Tinf = Tinf
        self.qW = qW
        self.qE = qE
        self.length = length
        self.area = area
        self.resolution = resolution

    @property
    def dx(self):
        if self.length and self.resolution:
            return self.length/self.resolution
        return None

class Parameters:
    """Définition des paramètres pour la résolution matricielle
    
    À définir : 
    
    1. a_e: coefficient est
    2. a_w: coefficient west
    3. S_u: source indépendante
    4. S_p: source dépendante"""

    def __init__(self,
                 a_e=None,
                 a_w=None,
                 S_u=None,
                 S_p=None):
        self.a_e = a_e
        self.a_w = a_w
        self.S_u = S_u
        self.S_p = S_p
        if (a_e!=None and a_w!=None and S_p!=None):
            self.a_p = a_e+a_w-S_p

def build_matrix (data,left,interior,right):
    """
    Cette fonction construit le système matriciel A·T = b associé au problème.
    - Chaque volume de contrôle donne lieu à une équation d’équilibre thermique.
    - Les conditions aux limites (températures imposées, convection) sont appliquées
      en modifiant directement la matrice et le second membre.
    
    Retourne :
    - matrixA : matrice des coefficients
    - matrixb : vecteur du second membre
    """

    matrixA = np.zeros((data.resolution,data.resolution),dtype=float)
    matrixb = np.zeros((data.resolution,1),dtype=float)

    for iPoint in range(data.resolution):

        if iPoint == 0:
            matrixA[iPoint][iPoint] = left.a_p
            matrixA[iPoint][iPoint+1] = -left.a_e
            matrixb[iPoint][0] = left.S_u

        elif iPoint == data.resolution-1:
            matrixA[iPoint][iPoint] = right.a_p
            matrixA[iPoint][iPoint-1] = -right.a_w
            matrixb[iPoint][0] = right.S_u

        else:
            matrixA[iPoint][iPoint] = interior.a_p
            matrixA[iPoint][iPoint-1] = -interior.a_w
            matrixA[iPoint][iPoint+1] = -interior.a_e
            matrixb[iPoint][0] = interior.S_u

    return matrixA,matrixb

def geometry (length=None, resolution=None):
    """Fonction qui détermine la position des points dans le domaine"""

    if length==None or resolution==None:
        print("Missing geometry information")
        return None
    dx = length/resolution
    nPoints = resolution+2
    points = np.zeros(nPoints,dtype=float)

    for iPoint in range(1,np.size(points)-1):
        value=dx/2+(iPoint-1)*dx
        points[iPoint] = value
    points[nPoints-1] = value+dx/2

    return points

def conduction_1D (data,left,interior,right):
    """Fonction qui prend en entrée les données du problème et les paramètres 
    requis pour la résolution matricielle.
    
    Paramètres:
    data (classe Data): données du probleme (résolution et la longueur 1D doivent définis)
    left (classe Parameters): paramètres de frontière ouest (a_e,a_w,S_u,S_p doivent être définis)
    interior (classe Parameters): paramètres à l'intérieur (a_e,a_w,S_u,S_p doivent être définis)
    right (classe Parameters): paramètres de frontière est (a_e,a_w,S_u,S_p doivent être définis) """

    
    A,b = build_matrix(data,left,interior,right)
    temperature = np.linalg.solve(A,b)
    return temperature

def plotGraph(data,numeric,analytic,title=None,fileName=None):
    xArrayNumeric = np.linspace(0+data.dx/2,data.length-data.dx/2,data.resolution)
    xArrayAnalytic = np.linspace(0,data.length,100)
    plt.plot(xArrayNumeric,numeric,"g+",label="numeric")#linestyle='dotted'
    plt.plot(xArrayAnalytic,analytic,label="analytic")
    plt.xlabel("position (m)")
    plt.ylabel("Temperature (C)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"{fileName}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

def error_L1(data,numeric,analytic):
    somme_L1 = sum(data.dx*abs(numeric[i]-analytic[i]) for i in range(data.resolution))
    return somme_L1[0]/data.length

def error_L1_graph(error,resolution,title=None,fileName=None):
    plt.plot(np.log(1/resolution),np.log(error),label="error")
    plt.xlabel("ln(h) (-) -- h~1/resolution")
    plt.ylabel("ln(Error) (-)")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(f"{fileName}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    ordre = (np.log(error[-1]/error[0])/np.log((1/resolution[-1])/(1/resolution[0])))
    return ordre


if __name__ == "__main__":

    ## Problem 1
    
    """
    Cas 1 : Convection-diffusion d'une propriété scalaire dans un domaine unidimensionnel

    ------------------------------------------------------------------

    On considère une barre métallique de longueur L = 1.0m, une masse volumique de 1kg/m³,
    une diffusivité de 0.1kg/m.s, une vitesse 1 de 0.1m/s et 2 de 2.5m/s.

    
    """
    

    def _parameters1(data):

        D = data.conduction/data.dx
        F = data.rho*data.velocity

        qu=0
        qp=0
        leftParameters = Parameters(a_e = D-F/2,
                                    a_w=0,
                                    S_u = (2*D+F)*data.phiW,
                                    S_p = -(2*D+F))

        rightParameters = Parameters(a_e=0,
                                    a_w=D+F/2,
                                    S_u = (2*D-F)*data.phiE,
                                    S_p = -(2*D-F))
        
        interiorParameters = Parameters(a_e=D-F/2,
                                    a_w=D+F/2,
                                    S_u=0,
                                    S_p=0)

        return leftParameters, interiorParameters, rightParameters
    
    data0 = Data(length=1, resolution=5, conduction=0.1, velocity=0.1, rho=1, phiW=1, phiE=0)
    parameters = _parameters1(data0)
    phi0 = conduction_1D(data0,parameters[0],parameters[1],parameters[2])

    data1 = Data(length=1, resolution=5, conduction=0.1, velocity=2.5, rho=1, phiW=1, phiE=0)
    parameters = _parameters1(data1)
    phi1 = conduction_1D(data1,parameters[0],parameters[1],parameters[2])

    data2 = Data(length=1, resolution=20, conduction=0.1, velocity=2.5, rho=1, phiW=1, phiE=0)
    parameters = _parameters1(data2)
    phi2 = conduction_1D(data2,parameters[0],parameters[1],parameters[2])





    ## Analyse probleme 1

    nStep=5 ## augmenter le nombre de step pour voir l'ordre de convergence
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)
    peclet = data0.rho*data0.velocity*data0.length/data0.conduction

    ## Polotting graph
    analytic = np.zeros(100,dtype=float)
    xArray = np.linspace(0,data0.length,100)
    for index,x in enumerate(xArray):
        analytic[index] = ((np.exp(peclet*(x/data0.length))-1)/(np.exp(peclet)-1))*(data0.phiE-data0.phiW)+data0.phiW

    title=f"Problem 5.1 - Convection-Diffusion, centered, 0.1m/s, {data0.resolution} points"
    fileName = "5_1_a"
    plotGraph(data0,phi0,analytic,title,fileName)

    # Error analysis
    for iStep in range(nStep):
        analytic = np.zeros(data0.resolution,dtype=float)
        parameters = _parameters1(data0)
        phi = conduction_1D(data0,parameters[0],parameters[1],parameters[2]) 
        xArray = np.linspace(data0.dx/2,data0.length-data0.dx/2,data0.resolution)
        for index,x in enumerate(xArray):
            analytic[index] = ((np.exp(peclet*(x/data0.length))-1)/(np.exp(peclet)-1))*(data0.phiE-data0.phiW)+data0.phiW
        err[iStep] = error_L1(data0,phi,analytic)
        resolution[iStep] = data0.resolution
        data0.resolution*=2
    
    title = "Problem 5.1 - Error analysis"
    fileName = "5_1_a_error"
    print("Ordre de l'erreur 5.1 à 2.5m/s et 5 points: ", error_L1_graph(err,resolution,title,fileName))
    print("---------------Commentaire----------------")
    print("------------------------------------------")


    nStep=5 ## augmenter le nombre de step pour voir l'ordre de convergence
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)
    peclet = data1.rho*data1.velocity*data1.length/data1.conduction

    analytic = np.zeros(100,dtype=float)
    xArray = np.linspace(0,data1.length,100)
    for index,x in enumerate(xArray):
        analytic[index] = ((np.exp(peclet*(x/data1.length))-1)/(np.exp(peclet)-1))*(data1.phiE-data1.phiW)+data1.phiW

    title=f"Problem 5.1 - Convection-Diffusion, centered, 2.5m/s, {data1.resolution} points"
    fileName = "5_1_b"
    plotGraph(data1,phi1,analytic,title,fileName)

    for iStep in range(nStep):
        analytic = np.zeros(data1.resolution,dtype=float)
        parameters = _parameters1(data1)
        phi = conduction_1D(data1,parameters[0],parameters[1],parameters[2]) 
        xArray = np.linspace(data1.dx/2,data1.length-data1.dx/2,data1.resolution)
        for index,x in enumerate(xArray):
            analytic[index] = ((np.exp(peclet*(x/data1.length))-1)/(np.exp(peclet)-1))*(data1.phiE-data1.phiW)+data1.phiW
        err[iStep] = error_L1(data1,phi,analytic)
        resolution[iStep] = data1.resolution
        data1.resolution*=2
    
    title = "Problem 5.1 - Error analysis"
    fileName = "5_1_b_error"
    print("Ordre de l'erreur 5.1 à 2.5m/s et 5 points: ", error_L1_graph(err,resolution,title,fileName))
    print("---------------Commentaire----------------")
    print("------------------------------------------")


    nStep=5 ## augmenter le nombre de step pour voir l'ordre de convergence
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)
    peclet = data2.rho*data2.velocity*data2.length/data2.conduction

    analytic = np.zeros(100,dtype=float)
    xArray = np.linspace(0,data2.length,100)
    for index,x in enumerate(xArray):
        analytic[index] = ((np.exp(peclet*(x/data2.length))-1)/(np.exp(peclet)-1))*(data2.phiE-data2.phiW)+data2.phiW

    title=f"Problem 5.1 - Convection-Diffusion, centered, 2.5m/s, {data2.resolution} points"
    fileName = "5_1_c"
    plotGraph(data2,phi2,analytic,title,fileName)

    for iStep in range(nStep):
        analytic = np.zeros(data2.resolution,dtype=float)
        parameters = _parameters1(data2)
        phi = conduction_1D(data2,parameters[0],parameters[1],parameters[2]) 
        xArray = np.linspace(data2.dx/2,data2.length-data2.dx/2,data2.resolution)
        for index,x in enumerate(xArray):
            analytic[index] = ((np.exp(peclet*(x/data2.length))-1)/(np.exp(peclet)-1))*(data2.phiE-data2.phiW)+data2.phiW
        err[iStep] = error_L1(data2,phi,analytic)
        resolution[iStep] = data2.resolution
        data2.resolution*=2
    
    title = "Problem 5.1 - Error analysis"
    fileName = "5_1_c_error"
    print("Ordre de l'erreur 5.1 à 2.5m/s et 20 points: ", error_L1_graph(err,resolution,title,fileName))
    print("---------------Commentaire----------------")
    print("------------------------------------------")



    ## Problem 2
    
    """
    Cas 1 : Convection-diffusion d'une propriété scalaire dans un domaine unidimensionnel

    ------------------------------------------------------------------

    On considère une barre métallique de longueur L = 1.0m, une masse volumique de 1kg/m³,
    une diffusivité de 0.1kg/m.s, une vitesse 1 de 0.1m/s et 2 de 2.5m/s.

    
    """
    

    def _parameters2(data):

        D = data.conduction/data.dx
        F = data.rho*data.velocity

        leftParameters = Parameters(a_e = D,
                                    a_w=0,
                                    S_u = (2*D+F)*data.phiW,
                                    S_p = -(2*D+F))

        rightParameters = Parameters(a_e=0,
                                    a_w=D+F,
                                    S_u = (2*D)*data.phiE,
                                    S_p = -(2*D))
        
        interiorParameters = Parameters(a_e=D+max(-F,0),
                                    a_w=D+max(F,0),
                                    S_u=0,
                                    S_p=0)

        return leftParameters, interiorParameters, rightParameters
    
    data0 = Data(length=1, resolution=5, conduction=0.1, velocity=0.1, rho=1, phiW=1, phiE=0)
    parameters = _parameters2(data0)
    phi0 = conduction_1D(data0,parameters[0],parameters[1],parameters[2])

    data1 = Data(length=1, resolution=5, conduction=0.1, velocity=2.5, rho=1, phiW=1, phiE=0)
    parameters = _parameters2(data1)
    phi1 = conduction_1D(data1,parameters[0],parameters[1],parameters[2])

    ## Analyse probleme 2

    nStep=10  ## augmenter le nombre de step pour voir l'ordre de convergence
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)
    peclet = data0.rho*data0.velocity*data0.length/data0.conduction

    ## Polotting graph
    analytic = np.zeros(100,dtype=float)
    xArray = np.linspace(0,data0.length,100)
    for index,x in enumerate(xArray):
        analytic[index] = ((np.exp(peclet*(x/data0.length))-1)/(np.exp(peclet)-1))*(data0.phiE-data0.phiW)+data0.phiW

    title=f"Problem 5.2 - Convection-Diffusion, centered, 0.1m/s, {data0.resolution} points"
    fileName = "5_2_a"
    plotGraph(data0,phi0,analytic,title,fileName)

    # Error analysis
    for iStep in range(nStep):
        analytic = np.zeros(data0.resolution,dtype=float)
        parameters = _parameters2(data0)
        phi = conduction_1D(data0,parameters[0],parameters[1],parameters[2]) 
        xArray = np.linspace(data0.dx/2,data0.length-data0.dx/2,data0.resolution)
        for index,x in enumerate(xArray):
            analytic[index] = ((np.exp(peclet*(x/data0.length))-1)/(np.exp(peclet)-1))*(data0.phiE-data0.phiW)+data0.phiW
        err[iStep] = error_L1(data0,phi,analytic)
        resolution[iStep] = data0.resolution
        data0.resolution*=2
    
    title = "Problem 5.2 - Error analysis"
    fileName = "5_2_a_error"
    print("Ordre de l'erreur 5.2 à 2.5m/s et 5 points: ", error_L1_graph(err,resolution,title,fileName))
    print("---------------Commentaire----------------")
    print("------------------------------------------")


    nStep=12 ## augmenter le nombre de step pour voir l'ordre de convergence
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)
    peclet = data1.rho*data1.velocity*data1.length/data1.conduction

    analytic = np.zeros(100,dtype=float)
    xArray = np.linspace(0,data1.length,100)
    for index,x in enumerate(xArray):
        analytic[index] = ((np.exp(peclet*(x/data1.length))-1)/(np.exp(peclet)-1))*(data1.phiE-data1.phiW)+data1.phiW

    title=f"Problem 5.2 - Convection-Diffusion, centered, 2.5m/s, {data1.resolution} points"
    fileName = "5_2_b"
    plotGraph(data1,phi1,analytic,title,fileName)

    for iStep in range(nStep):
        analytic = np.zeros(data1.resolution,dtype=float)
        parameters = _parameters2(data1)
        phi = conduction_1D(data1,parameters[0],parameters[1],parameters[2]) 
        xArray = np.linspace(data1.dx/2,data1.length-data1.dx/2,data1.resolution)
        for index,x in enumerate(xArray):
            analytic[index] = ((np.exp(peclet*(x/data1.length))-1)/(np.exp(peclet)-1))*(data1.phiE-data1.phiW)+data1.phiW
        err[iStep] = error_L1(data1,phi,analytic)
        resolution[iStep] = data1.resolution
        data1.resolution*=2
    
    title = "Problem 5.2 - Error analysis"
    fileName = "5_2_b_error"
    print("Ordre de l'erreur 5.2 à 2.5m/s et 5 points: ", error_L1_graph(err,resolution,title,fileName))
    print("---------------Commentaire----------------")
    print("------------------------------------------")