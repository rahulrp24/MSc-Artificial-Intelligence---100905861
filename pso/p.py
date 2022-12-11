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
    #Lb =  np.multiply(np.random.rand(0,1),0.1 * np.ones((1,noV)))
    # lb = 0 , ub = 1
    
    #Ub = 1 * np.ones((1,noV))
    
 
    Lb = []
    for i in range(noV):
        Lb.append(0.0001)
    
    

    Ub = []
    for i in range(noV):
        Ub.append(0.0099)
    
    #-----------------------------------------------------------#
#   Position[i,np.arange()]=Lb[i] + np.multiply((Ub[i] - Lb[i])*np.random.rand(np.size(Lb[i])))

#         pBest[i,np.arange()]=Position(i,np.arange())


    # for i in range(Position.shape[0]):
        
    #     for j in range(Position.shape[1]):
            
    #         Position[[i][j]]= (np.array(Lb[j]) + (np.array(Ub[j])-np.array(Lb[j])) * np.random.rand(np.size(Lb[j])))
    #         print(Position[np.arange(i,j)])
    #         pBest[np.arange(i,j)]=Position[np.arange(i,j)]
    # x=[]
    # for i in range(noP):
    #     for j in range(noV):
    #         m = np.array((i,j))
    #         x.append(m)
    #         Position[i][j] = (np.array(Lb[j]) + (np.array(Ub[j])-np.array(Lb[j])) * np.random.rand(np.size(Lb[j])))
    #         print(Position[i][j])
    #         pBest[i][j] = Position[i][j]

    lr=[]
    for i in range(noP):
        Position[i][0] = 0.0001 + (0.0009 - 0.0001) * np.random.rand(np.size(0.0001))
        lr.append(Position[i][0])
    print('Learning rate :',lr)

    drp=[]
    for i in range(noP):
        Position[i][1] = 0.3 + (0.7 - 0.3) * np.random.rand(np.size(0.3))
        drp.append(Position[i][1])
    print('dropout :',drp)

    # hidden=[]
    # for i in range(noP):
    #     Position[i][2] = 500 + (2500 - 500) * np.random.rand(np.size(500))
    #     hidden.append(Position[i][2])
    # print(hidden)

 
    #--------------------------------------------------------- #
    
    for l in range(Max_iteration):
        #---------  start at T1 ----- #
        #Calculate cost for each particle
        for i in range(len(Position)):
            # -- calculate fitness value of each particle --#
#[fitness_values, pBest_binary_solution(i,:)] = CostFunction(Position(i,:), training_in, training_label);
#deep learning evaluation
#digit recogniiton 0-9
#zn(i)=SimpleDeepLearning(ns(i,1),ns(i,2), trainDigitData, valDigitData);
#skin lesion
#zn(i)=Demo_TrainingFromScratch_skin_isc2(ns(i,1),ns(i,2), trainDigitData, valDigitData);

            fitness_values = PSO_nn(lr[i],drp[i],train,val)
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

                #print(Ub[0],Ub[1])
                #exit
                #-- checking Position in range of Up and Lb --#
#--- if the Position over the boundary upper then took upper value
#-- if the Position lower the boundary then took lower value
                if j ==0:
                    if Position[i,j] > (0.0009):
                        Position[i,j] = (0.0009)
                    if Position[i,j] < (0.0001):
                        Position[i,j] = (0.0001)

                elif j ==1:
                    if Position[i,j] > (0.7):
                        Position[i,j] = (0.7)
                    if Position[i,j] < (0.3):
                        Position[i,j] = (0.3)      

                elif j ==2:
                    if Position[i,j] > (500):
                        Position[i,j] = (500)
                    if Position[i,j] < (2500):
                        Position[i,j] = (2500)     
        #------------  end each iteration of PSO ----- #
        print(np.array([str(l),'_PSO']))
    
    return gBest_all,gBest_all_position
    
    