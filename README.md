# MSc-Artificial-Intelligence-project-100905861



For image caption generation

upload the file in google colab to run.

just need to run all the cells 


For emotion recognition 

setting up the environment

download and install anaconda 

download environment.yml file from    https://project-msc-ai.s3.eu-west-2.amazonaws.com/environment.yml

please copy the downloaded file in   c://users/username/          **username is the windows username**

open anaconda command prompt ( this is not the normal windows command prompt) and run the following code

command conda env install -f environment.yml

then activate the new environment -> conda activate project1


download the dataset from https://project-msc-ai.s3.eu-west-2.amazonaws.com/fer2013.zip 

extract the zip file 

copy the train and test folder inside the dataset folder inside the emotion folder

------ training -----
in the anaconda command prompt type " python train.py " to start the training process, if error occurs please check if the dataset path is correct 

------testing ------
after training using the anaconda command prompt navigate to the emotion folder and type " python emotion.py "  

to quit press " q "

 


----------------------------------------
##source env export > environment.yml
----------------------------------------
