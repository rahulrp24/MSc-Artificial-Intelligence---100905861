%% Implementation of Particle Swarm Optimization program %
%  by Worawut Srisukkham         6  December 15 -- %

function [gBest_all , gBest_all_position]= PSO_FS(noP, noV, Max_iteration, sequencesTrain, labelsTrain, sequencesValidation, labelsValidation)

%Initial Parameters for PSO
% w=2;                 %Inirtia weight
 w= 0.5;               %Inirtia weight
 c1= 1.5; %2;          % social learning rate
 c2= 1.5; % 2;         % individual learning rate   

%---- set Start Iteration for running each real iteration --%
Velocity=zeros(noP,noV);%Velocity vector
Position=zeros(noP,noV);%Position vector

%////////Cognitive component///////// 
pBestScore=zeros(noP);
pBest=zeros(noP,noV);
%////////////////////////////////////

%////////Social component///////////
% gBestScore=inf;
 % gBestScore= -inf; %-- using -inf due to my fitness function best fitness is higest value
gBest_all= -inf; %-- using -inf due to my fitness function best fitness is higest value
gBest_all_position=zeros(1,noV);

%///////////////////////////////////

%----------------------------------------------
Lb = 0.0001*ones(1,noV); %0.00001
Ub = 0.0005*ones(1,noV); %0.05
%-----------------------------------------------------------%

%Initialization
for i=1:size(Position,1) % For each particle

            Position(i,:) = Lb +(Ub-Lb).* rand(size(Lb));
       %--- initial value for pBest -- %
            pBest(i,:) = Position(i,:);
end
%--------------------------------------------------------- %

for l=1:Max_iteration
    %---------  start at T1 ----- %

    %Calculate cost for each particle
    for i=1:size(Position,1)  
       
          % -- calculate fitness value of each particle --%
          %[fitness_values, pBest_binary_solution(i,:)] = CostFunction(Position(i,:), training_in, training_label);
          %deep learning evaluation
          %digit recogniiton 0-9
          %zn(i)=SimpleDeepLearning(ns(i,1),ns(i,2), trainDigitData, valDigitData);
          %skin lesion
          %zn(i)=Demo_TrainingFromScratch_skin_isc2(ns(i,1),ns(i,2), trainDigitData, valDigitData);
          fitness_values = Demo_TrainingFromScratch_skin_ph2_updated2(abs(Position(i,1)),abs(Position(i,2)), sequencesTrain, labelsTrain, sequencesValidation, labelsValidation);
         
             %--- update pBest ---%  
          if(pBestScore(i) < fitness_values)
                pBestScore(i) = fitness_values;
                pBest(i,:)= Position(i,:);
          end   
			 %--- update gBest --%
          if (gBest_all < fitness_values)
                gBest_all = fitness_values;
                gBest_all_position = Position(i,:);
               
          end    

    end  %-- for i : Position(i,:)

 %-- moving and updating PSO position using formula of original PSO (Kennedy and Eberhart, 2003) -- %
 
 for i=1 : size(Position,1)
      for j=1 : size(Position,2)
	  

   %----------- according to traditional PSO ---- %
        Velocity(i,j) = w*Velocity(i,j) + c1*rand*(pBest(i,j)- Position(i,j)) + c2*rand*(gBest_all_position(j) - Position(i,j)); 
		Position(i,j) = Position(i,j) + Velocity(i,j);

		 
           %-- checking Position in range of Up and Lb --%
           %--- if the Position over the boundary upper then took upper value
           %-- if the Position lower the boundary then took lower value
		   
              if Position(i,j) > Ub(1,1)  %-- 5 -- %
                  Position(i,j) = Ub(1,1); 
              end
              if Position(i,j) < Lb(1,1)  %-- -5 -- %
                  Position(i,j) = Lb(1,1);
              end
   
    
      end  %-- j -- %
 end  %-- i -- %
 %------------  end each iteration of PSO ----- %
  
 disp({num2str(l), '_PSO'});

end   %-- iteration -- %

end  %-- end function PSO -- %



