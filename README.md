# AI PHYSICAL CHESS :chess_pawn: :eye:	

> An AI agent you can play with, physically, using computer vision technics.
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
  
  First is the Computer Vision part. That part incharges of understanding the given physical-board, live on camera. 
  It idenifys the board , the 64 squers of it, and the chess pieces structure. 
  
  Second is the AI agent part. That part incharges of generating the moves. 
  It gets the current position from the Computer Vision part, and detemnis what it believs to be the best move. 
------------------------------------------------------------------------
  
  The two parts combined -allow the user to play an AI agent, on a physical board. Thats a combination that one doesnt see regulary, as most AI agents can be played     againts only through a digital board.

  The project has an option of playing againts Stockfish-15 as well. 
  

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Tools Used :bulb:	
-  :eye:	Computer Vision part - 
  
    Python, Open CV, Keras, VGG19 model,  MediaPipe.


- :robot:	 AI agent part - 

  Python, python-chess library, AI algotythms




## game procces 
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3

#  - project structure 
## Computer Vision part:
  - crecollect Data from pysical board. 

  :mag_right:	-uploading images of boards with chess pieces on it.

  :mag_right:	-resizing images.

  :mag_right:	-converting images to a gray and blurred version - to get some image noise out for a better later edge ditecton.

  :mag_right:	-edge detection.

  :mag_right:	-strights lines detection, while classifing to HORIZONTAL and VERTICAL. 

  :mag_right:	-detecting lines that couldnt be found, and adding them. 

  :mag_right:	-finding lines intersection to determine the 64 squers.

  :mag_right:	-converting each 4 points that represent a square into an equilateral squere.
  
  :mag_right:	-manualy determines the piece on the square and classifies it to the relevent class file. 70% train, 15% validation, and another 15% for the test class.
  
   - train VGG19 model. 
   
   :toolbox:	-use the data collected in the previous phase to train, the last layer of a VGG19 model. 
   
   :toolbox:	-validate and test VGG19. 
   
   - activate VGG19 model to determine pieces's structure in a given position
    
 ## AI agent part:
 
   - evaluation functions.
   -A batch of evaluation function to determine the 'value' of a given chess position. 
   
   The valuation fuctions represent : 
   
   :chess_pawn:-location of pieces on board in a generic manner.
   
   :chess_pawn:-piece capture.
   
   :chess_pawn:-space ( attacked squers ) ocupied. 
   
   :chess_pawn:-pawn structur.
   
   :chess_pawn:-proximity to the enemy's king
   
   :chess_pawn:-attack squares in relation to defendor squares.
   
   :chess_pawn:-caslte.
   
   :chess_pawn:-left pieces with respect to strength.
   
   :chess_pawn:-bishop protector.
   
   :chess_pawn:-connected rooks.
   
   Its important to mention that these evaluation function are only a fruction from possible evaluation functions.
   
   
   
   - AI aglorythms - using the evaluation functions and determines best move. depth 3.
   
   :game_die:	-MINIMAX
   
   :game_die:	-ALPHA BETA PRUNING
   
   :game_die:	-QUIENSCENCE SEARCH
   
   :game_die:	-GENETIC ALGORYTHM
   
   - Opening Theory Book - pre maid desition tree for the opening phase. 
   
   :notebook:	-the AI agent uses that database of moves as long as possible. 
   
   - Stockfish-15
  
  :tropical_fish:	-The AI agent trains over Stockfish-15 in the GENETIC ALGORYTHM
   
   
   
  


## Screenshots :camera_flash:	
![Example screenshot](C:\Users\itama\PycharmProjects\CVCai\SHAY_2.jpeg)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup :hammer_and_wrench:	

- computer's camera

- download all files and change 'path's in code accordinly.

- non-trivial libraries needed:

    import cv2
    import numpy as np
    from keras import Model
    from keras.applications import VGG19
    from keras.callbacks import EarlyStopping
    from keras.layers import Flatten, Dense
    from keras.preprocessing.image import ImageDataGenerator
    from matplotlib import pyplot as plt
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located?

Proceed to describe how to install / setup one's local environment / get started with the project.

## Files :open_file_folder:	
- create_data.py -> 

creates the data using computer vision technics. classifies squares to diffrent classe folders.

- VGG-19_train ->

trains the last dense of VGG-19 on 'create_data.py' data


- VGG-19_activate -> 

classifies 64 squers into a board representation. 

- engin -> 

holds evaluation functions and AI algorithms.

There is an option for activate only that part ( without the computer vision part) 

- main -> 

holds the root code that operate the whole orchestra. 

- stockfish_15....zip ->

holds the 'brain' of stockfish

- VGGMODEL.h5 -> 

holds the saved VGG-19 model, trained with our dataset. saves resources as we dont need to train the model all over again each time. 

- test, train, val -> 

each file holds the classes of the pieces with manualy classified images( squers) 

- computer_data.bin ->

holds the 'Opening theory book' represented by a binary format. 


## Project Status :avocado:	
Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why.


## Room for Improvement  :weight_lifting_man:	
Include areas you believe need improvement / could be improved. Also add TODOs for future development.



Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1screen
- Feature to be added 2 



## Contact :email:	
Created by [@Itamar](https://www.linkedin.com/in/itamar-stollman) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
