# AI PHYSICAL CHESS :chess_pawn: :eye:	

> An AI agent you can play with, physically, using computer vision techniques.
> Live demo [_here_](https://www.example.com). <!-- If you have the project hosted somewhere, include the link here. -->

    
## Table of Contents  :crystal_ball:	
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)
<!-- * [License](#license) -->

## General Information :desktop_computer:	
- Provide general information about your project here.
    
  The project has 2 main parts.
  
  First is the Computer Vision part. That part is in charge of understanding the given physical board, live on camera. 
  It identifies the board, the 64 squares of it, and the chess pieces' structure. 
  
  Second is the AI agent part. That part is in charge of generating the moves. 
  It gets the current position from the Computer Vision part and generates what it believes to be the best move. 
------------------------------------------------------------------------
  
  The two parts combined -allow the user to play an AI agent, on a physical board. That's a combination that one doesn't see regularly, as most AI agents can be played     against only through a digital board.

  The project has an option of playing against Stockfish-15 as well. 
  

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Tools Used :bulb:	
-  :eye:	Computer Vision part - 
  
    Python, Open CV, Keras, VGG19 model,  MediaPipe.


- :robot:	 AI agent part - 

  Python, python-chess library, AI algorithms




## game process 
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3

#  - project structure 
## Computer Vision part:
  - collect Data from physical board. 

  :mag_right:	-uploading images of boards with chess pieces on it.

  :mag_right:	-resizing images.

  :mag_right:	-converting images to a gray and blurred version - to get some image noise out for a better later edge detection.

  :mag_right:	-edge detection.

  :mag_right:	-straight lines detection, while classifying to HORIZONTAL and VERTICAL. 

  :mag_right:	-detecting lines that couldn't be found, and adding them. 

  :mag_right:	-finding lines intersection to determine the 64  squares.

  :mag_right:	-converting each 4 points that represent a square into an equilateral square.
  
  :mag_right:	-manually determines the piece on the square and classifies it to the relevant class file. 70% train, 15% validation, and another 15% for the test class.
  
   - train VGG19 model. 
   
   :toolbox:	-use the data collected in the previous phase to train, the last layer of a VGG19 model. 
   
   :toolbox:	-validate and test VGG19. 
   
   - activate VGG19 model to determine pieces's structure in a given position
    
 ## AI agent part:
 
   - evaluation functions.
   -A batch of evaluation functions to determine the 'value' of a given chess position. 
   
   The valuation functions represent : 
   
   :chess_pawn:-location of pieces on board in a generic manner.
   
   :chess_pawn:-piece capture.
   
   :chess_pawn:-space ( attacked squers ) occupied. 
   
   :chess_pawn:-pawn structure.
   
   :chess_pawn:-proximity to the enemy's king
   
   :chess_pawn:-attack squares in relation to defendor squares.
   
   :chess_pawn:-castle.
   
   :chess_pawn:-left pieces with respect to strength.
   
   :chess_pawn:-bishop protector.
   
   :chess_pawn:-connected rooks.
   
   Its important to mention that these evaluation functions are only a fractionfrom possible evaluation functions.
   
   
   
   - AI algorithms- using the evaluation functions and determining the best move. depth 3.
   
   :game_die:	-MINIMAX
   
   :game_die:	-ALPHA BETA PRUNING
   
   :game_die:	-QUIESCENCE SEARCH
   
   :game_die:	-GENETIC ALGORITHM
   
   - Opening Theory Book - pre made decision tree for the opening phase. 
   
   :notebook:	-the AI agent uses that database of moves as long as possible. 
   
   - Stockfish-15
  
  :tropical_fish:	-The AI agent trains over Stockfish-15 in the GENETIC ALGORITHM
   
   
   
  


## Screenshots :camera_flash:	
![Example screenshot](C:\Users\itama\PycharmProjects\CVCai\SHAY_2.jpeg)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup :hammer_and_wrench:	

- computer's camera

- download all files and change 'path's in code accordingly.

- non-trivial libraries needed:

        import cv2
        import numpy as np
        from tensorflow import keras
        from keras import Model
        from keras.applications import VGG19
        from keras.callbacks import EarlyStopping
        from keras.layers import Flatten, Dense
        from keras.preprocessing.image import ImageDataGenerator
        from matplotlib import pyplot as plt
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        
        import chess.engine
        import mediapipe as mp



## Files :open_file_folder:	
- create_data.py -> 

creates the data using computer vision techniques. classifies squares to differentclasse folders.

- VGG-19_train ->

trains the last dense of VGG-19 on 'create_data.py' data


- VGG-19_activate -> 

classifies 64 squares into a board representation. 

- engin -> 

holds evaluation functions and AI algorithms.

There is an option for activate only that part ( without the computer vision part) 

- main -> 

holds the root code that operates the whole orchestra. 

- stockfish_15....zip ->

holds the 'brain' of stockfish

- VGGMODEL.h5 -> 

holds the saved VGG-19 model, trained with our dataset. saves resources as we dont need to train the model all over again each time. 

- test, train, val -> 

each file holds the classes of the pieces with manually classified images( squers) 

- computer_data.bin ->

holds the 'Opening theory book' represented by a binary format. 


## Project Status :avocado:	
Project is _almost finished_ -  _small improvements need to be made for a smooth user experience,_

_As well as there are some changes that need to made for a cleaner code and better usage of raw materials_


## Room for Improvement  :weight_lifting_man:	
Include areas you believe need improvement / could be improved. Also add TODOs for future development.



Room for improvement in Computer Vision part:

- train VGG-19 over more images for better accuracy. 

- train VGG-19 over more chess boards and pieces sets. 

-explore a largerr range of shooting angles.

- increase accuracy of board detection

- add statistical approach to allow a better piece recognition



Room for improvement in AI agent part:

- code more evaluation functions

- run longer and better GENETIC ALGORITHM
 
- train GENETIC ALGORITHM on human players as well as on various AI agents

- use Neural Networks approach

- make the structure faster by using state of the art approaches.

:bust_in_silhouette:	Add a robotic hand that will enable a more realistic game against the AI agent. 


## Contact :email:	
Created by [@Itamar](https://www.linkedin.com/in/itamar-stollman) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
