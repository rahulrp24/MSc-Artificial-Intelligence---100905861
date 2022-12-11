

import numpy as np
import math




    ## Implementation of Particle Swarm Optimization program #
#  by Worawut Srisukkham         6  December 15 -- #
    
    

def PSO(noP,noV,Max_iteration,train,val):


    #Initial Parameters for PSO
                #Inirtia weight
    w=0.5

    
    c1=1.5

    
    c2=1.5

    
    #---- set Start Iteration for running each real iteration --#
    Velocity=(np.zeros((noP,noV)))

    
    Position=(np.zeros((noP,noV)))

    
    #////////Cognitive component/////////
    pBestScore=np.zeros(noP)

    pBest=(np.zeros((noP,noV)))

    #////////////////////////////////////
    
    #////////Social component///////////
# gBestScore=inf;
 # gBestScore= -inf; #-- using -inf due to my fitness function best fitness is higest value
    gBest_all=- math.inf

    
    gBest_all_position=(np.zeros((1,noV)))

    #///////////////////////////////////
    
    #----------------------------------------------
    Lb=np.dot(0.0001,np.ones((1,noV)))

    
    Ub=np.dot(0.0005,np.ones((1,noV)))

    
    #-----------------------------------------------------------#
    
    #Initialization
    for i in np.arange(0,np.size(Position,1)).reshape(-1):
        Position[i,np.arange()]=Lb[i] + np.multiply((Ub[i] - Lb[i])*np.random.rand(np.size(Lb[i])))

        pBest[i,np.arange()]=Position(i,np.arange())

    
    
    


    
    #--------------------------------------------------------- #
    
    for l in np.arange(0,Max_iteration).reshape(-1):
        #---------  start at T1 ----- #
        #Calculate cost for each particle
        for i in np.arange(0,Position.shape[1-1]).reshape(-1):
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
        print(np.array([str(l),'_PSO']))
    
    return gBest_all,gBest_all_position
    
    