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
                 heatGeneration=None,
                 T1=None,
                 T2=None,
                 Tinf=None,
                 q1=None,
                 q2=None,
                 length=None,
                 area=None,
                 resolution=None):
        self.conduction = conduction
        self.convection = convection
        self.heatGeneration = heatGeneration
        self.T1 = T1
        self.T2 = T2
        self.Tinf = Tinf
        self.q1 = q1
        self.q2 = q2
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

def error_L1(data,numeric,analytic,plot=False,string=None):
    if plot == True:
        points = geometry(data.length,data.resolution)
        plt.plot(points[1:-1],numeric,"g+",label="numeric")
        plt.plot(points[1:-1],analytic,label="analytic")
        plt.xlabel("position (m)")
        plt.ylabel("Temperature (C)")
        plt.title(string)
        plt.legend()
        plt.grid()
        plt.savefig(f"{string}.png", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    somme_L1 = sum(data.dx*abs(numeric[i]-analytic[i]) for i in range(data.resolution))
    return somme_L1[0]/data.length

def error_L1_graph(error,resolution,string=None):
    plt.plot(np.log(1/resolution),np.log(error),label="error")
    plt.xlabel("ln(h) (-) -- h~1/resolution")
    plt.ylabel("ln(Error) (-)")
    plt.title(string)
    plt.grid()
    plt.legend()
    plt.savefig(f"{string}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    ordre = (np.log(error[-1]/error[0])/np.log((1/resolution[-1])/(1/resolution[0])))
    return ordre


if __name__ == "__main__":

    ## Problem 1
    
    """
    Cas 1 : Conduction thermique 1D sans source de chaleur
    ---------------------------------------------------------
    On considère une barre isolée thermiquement sur ses faces latérales,
    de longueur L = 0.5 m, dont les extrémités sont maintenues à des
    températures constantes :
    - T(0) = 100 °C
    - T(L) = 500 °C
    
    Le problème est gouverné par l’équation de conduction stationnaire
    sans source volumique (q = 0). On cherche la répartition de température
    en régime permanent dans la barre.
    
    Données :
    - Conductivité thermique : k = 1000 W/m.K
    - Aire de section : A = 0.01 m²
    
    """
    

    def _parameters1(data):
        qu=0
        qp=0
        leftParameters = Parameters(a_e = data.conduction*data.area/data.dx,
                                    a_w=0,
                                    S_u = qu + 2*data.conduction*data.area*data.T1/data.dx,
                                    S_p = qp - 2*data.conduction*data.area/data.dx)

        rightParameters = Parameters(a_e=0,
                                    a_w=data.conduction*data.area/data.dx,
                                    S_u = qu + 2*data.conduction*data.area*data.T2/data.dx,
                                    S_p = qp - 2*data.conduction*data.area/data.dx)
        
        interiorParameters = Parameters(a_e=data.conduction*data.area/data.dx,
                                    a_w=data.conduction*data.area/data.dx,
                                    S_u=qu,
                                    S_p=qp)

        return leftParameters, interiorParameters, rightParameters
    
    data = Data(length=0.5, resolution=5, T1=100, T2=500, area=10e-2, conduction=1000)
    parameters = _parameters1(data)
    conduction_1D(data,parameters[0],parameters[1],parameters[2])

    ## Analyse probleme 1

    nStep=10
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)

    plot=True
    title=f"Problem 4.1 - Temperature comparaison,{data.resolution*2} points"
    for iStep in range(nStep):
        data.resolution*=2
        analytic = np.zeros(data.resolution,dtype=float)
        parameters = _parameters1(data)
        temperature = conduction_1D(data,parameters[0],parameters[1],parameters[2]) 
        for iPoint in range(np.size(analytic)):
            analytic[iPoint] = 800*(iPoint*data.dx+data.dx/2)+100
        err[iStep] = error_L1(data,temperature,analytic,plot,title)
        resolution[iStep] = data.resolution
        plot=False
    
    title = "Problem 4.1 - Error analysis"
    print("Ordre de l'erreur 4.1 : ", error_L1_graph(err,resolution,title))
    print("---------------Commentaire----------------")
    print("Ici, l'ordre de l'erreur semble étrange ; il indique que l'erreur diminue lorsque la taille du pas augmente." \
    "Ce comportement étrange est potentiellement dû au fait que le résultat obtenu est exactement approximé par la méthode" \
    "des volumes finis. En effet, comme l'erreur est minuscule (10e-20 à 10e-30), il s'agit en fait de l'erreur en virgule flottante." \
    "Ainsi, plus on a de points (plus on réduit la taille des éléments), plus l'erreur s'accumule.")
    print("------------------------------------------")



    ## Problem 2:
        
    """
    Cas 2 : Conduction thermique 1D avec génération uniforme de chaleur
    ------------------------------------------------------------------
    On considère une plaque de grande dimension (dans les directions y et z),
    d’épaisseur L = 0.02 m. On suppose que le transfert de chaleur
    ne se fait que selon l’axe x.
    
    Les faces sont maintenues à des températures constantes :
    - T(0) = 100 °C
    - T(L) = 200 °C
    
    Le matériau possède une conductivité thermique constante :
    - k = 0.5 W/m.K
    
    Une source interne de chaleur uniforme est présente :
    - q = 1000 kW/m³ = 1.0 × 10⁶ W/m³
    
    Hypothèses :
    - Régime permanent (état stationnaire)
    - Problème purement 1D
    - Pas de flux de chaleur latéral (plaque très large)
    """
        
    def _parameters2(data):

        qu=data.area*data.heatGeneration*data.dx
        qp=0
        leftParameters = Parameters(a_e = data.conduction*data.area/data.dx,
                                    a_w=0,
                                    S_u = qu + 2*data.conduction*data.area*data.T1/data.dx,
                                    S_p = qp - 2*data.conduction*data.area/data.dx)

        rightParameters = Parameters(a_e=0,
                                    a_w=data.conduction*data.area/data.dx,
                                    S_u = qu + 2*data.conduction*data.area*data.T2/data.dx,
                                    S_p = qp - 2*data.conduction*data.area/data.dx)
        
        interiorParameters = Parameters(a_e=data.conduction*data.area/data.dx,
                                    a_w=data.conduction*data.area/data.dx,
                                    S_u=qu,
                                    S_p=qp)
        
        return leftParameters, interiorParameters, rightParameters
    
    data = Data(length=0.02, resolution=5, conduction=0.5, heatGeneration=1000000,T1=100, T2=200, area=1)
    parameters = _parameters2(data)
    conduction_1D(data,parameters[0],parameters[1],parameters[2])

    ## Analyse probleme 2

    nStep=5
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)

    plot=True
    title=f"Problem 4.2 - Temperature comparaison,{data.resolution*2} points"
    for iStep in range(nStep):
        data.resolution*=2
        analytic = np.zeros(data.resolution,dtype=float)
        parameters = _parameters2(data)
        temperature = conduction_1D(data,parameters[0],parameters[1],parameters[2])
        for iPoint in range(np.size(analytic)):
            x=iPoint*data.dx+data.dx/2
            analytic[iPoint] = (((data.T2-data.T1)/data.length)+(data.heatGeneration*(data.length-x)/(2*data.conduction)))*x+data.T1
        err[iStep] = error_L1(data,temperature,analytic,plot,title)
        resolution[iStep] = data.resolution
        plot=False

    title = "Problem 4.2 - Error analysis"
    print("Ordre de l'erreur 4.2: ", error_L1_graph(err,resolution,title))

    ## Problem 3:
        
    """
    Cas 3 : Refroidissement d’une ailette cylindrique par convection
    ----------------------------------------------------------------
    On considère une ailette cylindrique de section constante (aire A),
    soumise à un transfert de chaleur par convection tout au long de sa longueur.
    
    Conditions aux limites :
    - Base de l’ailette maintenue à une température fixe : T(0) = 100 °C
    - Extrémité libre (x = L) supposée isolée (pas de flux thermique)
    - Température ambiante : T∞ = 20 °C
    
    Caractéristiques physiques :
    - Géométrie cylindrique avec aire de section A (constante)
    - Échanges thermiques avec l’air ambiant par convection
    - La conduction le long de l’ailette est 1D
    
    """
    
    def _parameters3(data):
        n2 = 25
        qu=n2*data.Tinf*data.dx
        qp=-n2*data.dx
        leftParameters = Parameters(a_e=1/data.dx,
                                    a_w=0,
                                    S_u=qu + 2*data.T1/data.dx,
                                    S_p=qp -2/data.dx)
        
        rightParameters = Parameters(a_e=0,
                                    a_w=1/data.dx,
                                    S_u=qu,
                                    S_p=qp)
        
        interiorParameters = Parameters(a_e=1/data.dx,
                                    a_w=1/data.dx,
                                    S_u=qu,
                                    S_p=qp)
        
        return leftParameters, interiorParameters, rightParameters

    data = Data(length=1, resolution=5,Tinf=20,T1=100,q2=0)
    parameters = _parameters3(data)
    conduction_1D(data,parameters[0],parameters[1],parameters[2])

    ## Analyse probleme 3

    nStep=5
    n=np.sqrt(25)
    err = np.zeros(nStep)
    resolution = np.zeros(nStep)

    plot=True
    title=f"Problem 4.3 - Temperature comparaison,{data.resolution*2} points"
    for iStep in range(nStep):
        data.resolution*=2
        analytic = np.zeros(data.resolution,dtype=float)
        parameters = _parameters3(data)
        temperature = conduction_1D(data,parameters[0],parameters[1],parameters[2])
        for iPoint in range(np.size(analytic)):
            x=iPoint*data.dx+data.dx/2
            analytic[iPoint] = (np.cosh(n*(data.length-x))/np.cosh(n*data.length))*(data.T1-data.Tinf)+data.Tinf
        err[iStep] = error_L1(data,temperature,analytic,plot,title)
        resolution[iStep] = data.resolution
        plot=False

    title = "Problem 4.3 - Error analysis"
    print("Ordre de l'erreur 4.3: ", error_L1_graph(err,resolution,title))