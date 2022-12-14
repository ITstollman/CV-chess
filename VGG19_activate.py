import os
import numpy as np
from keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt

from tensorflow import keras
import create_data

model2 = keras.models.load_model("VGGMODEL.h5")

print(model2)
board = []

# path = 'C://Users//itama//PycharmProjects//yael_checker//train//49p'

files = ["48b", "48k", "48kn", "48p", "48q", "48r",
         "49b", "49k", "49kn", "49p", "49q", "49r",
         "empty", 'UNKNOWN']

'''
generates a board from image of board(via path) 

@:param path -> path of the image( board position) 
@:return board -> list of sqrs, represents the board
'''


def generate_board(path):
    known = False
    for img in (os.listdir(path)):

        img = load_img(path + "/" + img, target_size=(224, 224))
        plt.imshow(img)
        plt.show()
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model2.predict(images, batch_size=1)
        print('pred', pred)

        # check for the class category
        for i in range(12):
            if pred[0][i] > 0.5:
                print(files[i])
                board.append(files[i])
                known = True
                break

        if not known:
            board.append(files[13])

    print(board)

# create_data.draw_move(img, squares[20], squares[10])
