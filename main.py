import os
import chess
import cv2
import mediapipe as mp
import time

# Initializing the Model
import VGG19_activate
from create_data import draw_move
from engin import game_loop_human, get_random_DNA, ai_move, print_board

#  TODO - use table in 'engin.py' file. there is a usage of similar tables as only one table is needed.
board_notation = {1: 'a1', 2: 'b2', 3: 'c3', 4: 'd4', 5: 'e5', 6: 'f6', 7: 'g7', 8: 'h8',
                  9: 'a2', 10: 'b2', 11: 'c2', 12: 'd2', 13: 'e2', 14: 'f2', 15: 'g2', 16: 'h2',
                  17: 'a3', 18: 'b3', 19: 'c3', 20: 'd3', 21: 'e3', 22: 'f3', 23: 'g3', 24: 'h3',
                  25: 'a4', 26: 'b4', 27: 'c4', 28: 'd4', 29: 'e4', 30: 'f4', 31: 'g4', 32: 'h4',
                  33: 'a5', 34: 'b5', 35: 'c5', 36: 'd5', 37: 'e5', 38: 'f5', 39: 'g5', 40: 'h5',
                  41: 'a6', 42: 'b6', 43: 'c6', 44: 'd6', 45: 'e6', 46: 'f6', 47: 'g6', 48: 'h6',
                  49: 'a7', 50: 'b7', 51: 'c7', 52: 'd7', 53: 'e7', 54: 'f7', 55: 'g7', 56: 'h7',
                  57: 'a8', 58: 'b8', 59: 'c8', 60: 'd8', 61: 'e8', 62: 'f8', 63: 'g8', 64: 'h8'}


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

'''
checks if a hand appeared on screen for a while.
represents a move.

@:return True if a hand appeared on screen for a while 
'''


def hand_move():
    hand1 = False
    hand2 = False

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        #  TODO - use the various mpHands capabilities to distinct different hand gestures

        if results.multi_hand_landmarks:
            hand1 = True

        if not results.multi_hand_landmarks and hand1:
            hand2 = True

        if hand1 and hand2:
            return True


'''
checks if a hand appeared on screen for a while.
represents a move.

@:param path -> path for an image of the current board
@:param prev_board -> a list of sqrs, represents the board, before last move.
@:return 0 if the game os over due to check mate
'''


def human_move(path, prev_board):
    print('THATS A MOVE')
    time.sleep(3)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(path, f'current_board.jpg'), frame)

    new_board = VGG19_activate.generate_board(path)
    new_move, from_sqr, to_sqr = board_changes(prev_board, new_board)

    move = chess.Move.from_uci(new_move)
    if move in board.legal_moves:
        board.push(move)
        print_board(board)
        draw_move(frame, from_sqr, to_sqr)

    if board.is_checkmate():
        print("GAME OVER - HUMAN WON THE GAME")
        return 0


'''
checks if a hand appeared on screen for a while.
represents a move.

@:param random_DNA -> the DNA of the ai
@:param path -> path for an image of the current board
@:param stockfish_or_engin -> 'stockfish' if stockfish ai is asked. engin otherwise.
@:return new_board -> a list of sqrs, represents the board after ai played
'''


def ai_move_(random_DNA, path, stockfish_or_engin):
    ret, frame = cap.read()
    start_time = time.time()

    if stockfish_or_engin == 'engin':

        BLACK_AI_move = ai_move(board, chess.BLACK, 3,
                                random_DNA["num_space_M"],
                                random_DNA["num_capture_M"],
                                random_DNA["num_pawn_structure_M"],
                                random_DNA["num_connected_rooks_M"],
                                random_DNA["num_enemy_king_magnet_M"],
                                random_DNA["num_the_defending_bishop_M"],
                                random_DNA["num_defending_vs_attacking_M"])
    else:
        engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\itama\\Downloads\\stockfish.exe")
        BLACK_AI_move = engine.play(board, chess.engine.Limit(time=0.3))

    end_time = time.time()
    print(end_time - start_time)
    print("BLACK_AI_move", BLACK_AI_move)
    board.push(BLACK_AI_move)
    print_board(board)

    from_sqr = BLACK_AI_move[:2]
    to_sqr = BLACK_AI_move[3:]
    from_sqr_number = 0
    to_sqr_number = 0

    for key, value in board_notation.items():
        if value == from_sqr:
            from_sqr_number = key
        if value == to_sqr:
            to_sqr_number = key

    draw_move(frame, from_sqr_number, to_sqr_number)

    new_board = VGG19_activate.generate_board(path)
    if board.is_checkmate():
        print("GAME OVER - COMPUTER WON THE GAME")
        return 0
    return new_board


'''
returns the played move

@:param -> prev_board as a list of materials 
@:param -> new_board as a list of materials 
@:return -> the played move, in 'letter number letter number' sequence, e.g a2a4 
            and from_sqr, to_sqr (numbers representation)
'''


def board_changes(prev_board, new_board):
    from_sqr = 0
    to_sqr = 0
    for i in range(63):
        if prev_board[i] != 'empty' and new_board[i] == 'empty':
            from_sqr = i
        elif prev_board[i] != new_board[i]:
            to_sqr = i
    return board_notation[from_sqr + 1] + board_notation[to_sqr + 1], from_sqr, to_sqr


cap = cv2.VideoCapture(0)
capture_num = 0

path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker'

prev_board = []
board = chess.Board()

random_DNA = get_random_DNA(1000)
print("GENERATED RANDOMLY ->  ", random_DNA)
human_to_play = True
stockfish_or_engin = 'stockfish'
while True:

    if hand_move():
        if human_to_play:
            if human_move(path, prev_board) == 0:
                break
            human_to_play = False
        else:
            # computer to play
            if ai_move_(random_DNA, path, stockfish_or_engin) == 0:
                break
            else:
                prev_board = ai_move_(random_DNA, path, stockfish_or_engin)
            human_to_play = True

    # if cv2.waitKey(1) & 0xFF == ord('s') and hand:
    #     print('SAVE')
    #     hand = False

print("game over")
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
