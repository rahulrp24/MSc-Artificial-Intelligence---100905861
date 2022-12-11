## Implementation of Particle Swarm Optimization program #
#  by Worawut Srisukkham         6  December 15 -- #

import numpy as np
import math
from pso_nn import *   
def PSO(noP,noV,Max_iteration,train,val): 
    #Initial Parameters for PSO
# w=2;                 #Inirtia weight
    w = 0.5
    
    c1 = 1.5
    
    c2 = 1.5
    
    #---- set Start Iteration for running each real iteration --#
    Velocity = np.zeros((noP,noV))
    
    Position = np.zeros((noP,noV))
    
    #////////Cognitive component/////////
    pBestScore = np.zeros((noP))
    pBest = np.zeros((noP,noV))
    #////////////////////////////////////
    
    #////////Social component///////////
# gBestScore=inf;
# gBestScore= -inf; #-- using -inf due to my fitness function best fitness is higest value
    gBest_all = - math.inf
    
    gBest_all_position = np.zeros((1,noV))
    #///////////////////////////////////
    
    #----------------------------------------------
    Lb =  np.ones((1,noV))
    # lb = 0 , ub = 1
    
    Ub = 1 * np.ones((1,noV))
    
 
    # Lb = []
    # for i in range(noV):
    #     Lb.append(0.1)
    
    

    # Ub = []
    # for i in range(noV):
    #     Ub.append(1)
    
    #-----------------------------------------------------------#


    # for i in np.arange(noP):
    #     position.extend()

    #Initialization
    # position = []
    # for i in np.arange(noP):
    #     if Lb[i].any() < Ub[i].any() :
    #                 position.extend(np.random.randint(Lb[i], Ub[i] + 1, 1, dtype=int))
    #     elif Lb[i].any() == Ub[i].any():
    #                 position.extend(np.array([Lb[i]], dtype=int))
        
    #     else:
    #         assert False
    
    #--------------------------------------------------------- #
    
    for l in np.arange(0,Max_iteration).reshape(-1):
        #---------  start at T1 ----- #
        #Calculate cost for each particle
        for i in np.arange(0,Position.shape[0]).reshape(-1):
            # -- calculate fitness value of each particle --#
#[fitness_values, pBest_binary_solution(i,:)] = CostFunction(Position(i,:), training_in, training_label);
#deep learning evaluation
#digit recogniiton 0-9
#zn(i)=SimpleDeepLearning(ns(i,1),ns(i,2), trainDigitData, valDigitData);
#skin lesion
#zn(i)=Demo_TrainingFromScratch_skin_isc2(ns(i,1),ns(i,2), trainDigitData, valDigitData);

            fitness_values = PSO_nn(abs(Position[i,1]),train,val)
            #--- update pBest ---#
            if (pBestScore[i] < fitness_values):
                pBestScore[i] = fitness_values
                pBest[i,:] = Position[i,:]
            #--- update gBest --#
            if (gBest_all < fitness_values):
                gBest_all = fitness_values
                gBest_all_position = Position[i,:]
        #-- moving and updating PSO position using formula of original PSO (Kennedy and Eberhart, 2003) -- #
        for i in np.arange(0,Position.shape[1-1]).reshape(-1):
            for j in np.arange(0,Position.shape[2-1]).reshape(-1):
                #----------- according to traditional PSO ---- #
                Velocity[i,j] = w * Velocity[i,j] + c1 * (pBest[i,j] - Position[i,j]) + c2  * (gBest_all_position[j] - Position[i,j])
                Position[i,j] = Position[i,j] + Velocity[i,j]
                #-- checking Position in range of Up and Lb --#
#--- if the Position over the boundary upper then took upper value
#-- if the Position lower the boundary then took lower value
                if Position[i,j] > Ub[0,1]:
                    Position[i,j] = Ub[0,1]
                if Position[i,j] < Lb[0,1]:
                    Position[i,j] = Lb[0,1]
        #------------  end each iteration of PSO ----- #
        print(np.array([str(l),'_PSO']),Ub[0,1],Lb[0,1],Position[i,j])

    
    return gBest_all,gBest_all_position
    
    