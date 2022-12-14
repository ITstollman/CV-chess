"""
--------------------------------------------------
HELLO AND WELCOME TO 'AI CHESS' BY ITAMAR STOLLMAN
--------------------------------------------------

INSTRUCTIONS->

TO ACTIVATE THE PROGRAM, YOU WILL NEED TO DO SOME FEW THINGS:

1. MAKE SURE YOU ARE ABLE TO IMPORT :
    chess.polyglot
    chess.engine

2. DOWNLOAD computer_data file and save it in the project files
3. DOWNLOAD stockfish file and save it in the project files

4. GO TO theory_book FUNCTION AND PLACE THE COMPLETE PATH AT THE RELEVANT PLACE
5. GO TO game_loop FUNCTION AND PLACE THE COMPLETE PATH AT THE RELEVANT PLACE

THATS IT! YOU ARE READY TO ACTIVATE THE PROGRAM.
--------------------------------------------------

AFTER STARTING THE PROGRAM, YOU WILL HAVE A FEW OPTION FOR MESSING AROUND THE PROGRAM.

******************************************************************************
BY TYPING 1 :

YOU WILL GET OT PLAY WITH A RANDOM-DNA AI CHESS.

WHEN YOUR TURN COMES, TYPE A MOVE IN THE FOLLOWING FORMAT: a2a4 FOR EXAMPLE.
******************************************************************************
BY TYPING 2 :

YOU WILL NEED TO TYPE HOW MANY GAMES YOU WOULD LIKE TO WATCH
BETWEEN A RANDOM-DNA CHESS VS STOCKFISH-15

POPCORN IS HIGHLY RECOMMENDED

******************************************************************************

"""

import random
import time

import chess.polyglot

import chess.engine
import numpy

NUM_OF_GAMES = 1

PAWN = 1
BISHOP = 2
KNIGHT = 3
ROOK = 4
QUEEN = 5
KING = 6

UNICODE_PIECE_SYMBOLS = {
    'R': '♖', "r": "♜",
    'N': "♘", "n": "♞",
    'B': "♗", "b": "♝",
    'Q': "♕", "q": "♛",
    'K': "♔", "k": "♚",
    'P': "♙", "p": "♟",
}

pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 35, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -30, -10, -10, -30, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]
rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]
queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [

    20, 200, 10, 0, 0, 10, 200, 20,
    -20, -20, -20, -20, -20, -20, -20, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
]

fixed_board = [1, 2, 3, 4, 5, 6, 7, 8,
               9, 10, 11, 12, 13, 14, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24,
               25, 26, 27, 28, 29, 30, 31, 32,
               33, 34, 35, 36, 37, 38, 39, 40,
               41, 42, 43, 44, 45, 46, 47, 48,
               49, 50, 51, 52, 53, 54, 55, 56,
               57, 58, 59, 60, 61, 62, 63, 64]

piece_value = {1: 1, 2: 3.15, 3: 3, 4: 5, 5: 9, 6: 1}

'''
a game between engine ai and stockfish-15

@:param DNA->space_M_BLACK, capture_M_BLACK, pawn_structure_M_BLACK, connected_rooks_M_BLACK,
              enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK, defending_vs_attacking_M_BLACK        
@:return "WINNER": the winner, "num_of_moves": num_of_moves, "ave time per move" : ave time per move
'''


def game_loop(space_M_BLACK,
              capture_M_BLACK, pawn_structure_M_BLACK, connected_rooks_M_BLACK,
              enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK, defending_vs_attacking_M_BLACK):
    board = chess.Board()

    num_of_moves = 0
    print_board(board)
    total_time = 0

    while not board.is_game_over(claim_draw=True):

        try:

            engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\itama\\Downloads\\stockfish.exe")
            move = engine.play(board, chess.engine.Limit(time=0.3))
            board.push(move.move)
        except:

            if total_time == 0:
                total_time = 1
            if num_of_moves == 0:
                num_of_moves = 1

            return {"WINNER": chess.WHITE, "num_of_moves": num_of_moves,
                    "ave time per move": total_time / num_of_moves / 2}
        depth = 3

        print("WHITE STOCKFISH AI played:", move.move)
        print_board(board)

        if board.is_checkmate():
            print("GAME OVER - WHITE WON THE GAME")
            print({"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                   "ave time per move": total_time / num_of_moves / 2})

            return {"WINNER": chess.WHITE, "num_of_moves": count_moves(num_of_moves),
                    "ave time per move": total_time / num_of_moves / 2}

        start_time = time.time()

        BLACK_AI_move = ai_move(board, chess.BLACK, depth, space_M_BLACK, capture_M_BLACK, pawn_structure_M_BLACK,
                                connected_rooks_M_BLACK,
                                enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK, defending_vs_attacking_M_BLACK)

        end_time = time.time()
        total_time += end_time - start_time
        print(total_time)

        print("BLACK_AI_move", BLACK_AI_move)
        board.push(BLACK_AI_move)
        print("-------------------")
        print("BLACK AI  played", BLACK_AI_move)
        print_board(board)
        num_of_moves = num_of_moves + 1
        print("---------count_moves----------", count_moves(num_of_moves))

        if board.is_checkmate():
            print("GAME OVER - BLACK WON THE GAME")
            print(total_time)
            print({"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                   "ave time per move": total_time / num_of_moves / 2})

            return {"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                    "ave time per move": total_time / num_of_moves / 2}


'''
a game between manual human and stockfish-15

@:param DNA->space_M_BLACK, capture_M_BLACK, pawn_structure_M_BLACK, connected_rooks_M_BLACK,
              enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK, defending_vs_attacking_M_BLACK
              
@:return "WINNER": the winner, "num_of_moves": num_of_moves, "ave time per move" : ave time per move
'''


def game_loop_human(space_M_BLACK,
                    capture_M_BLACK, pawn_structure_M_BLACK, connected_rooks_M_BLACK,
                    enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK, defending_vs_attacking_M_BLACK):
    board = chess.Board()

    num_of_moves = 0
    print_board(board)
    total_time = 0

    while not board.is_game_over(claim_draw=True):

        input_move = input("HUMAN, Enter your move: ")
        move = chess.Move.from_uci(input_move)
        if move in board.legal_moves:
            board.push(move)
        else:
            #  TODO - to add a perpetual condition
            #         to add an exception for an invalid typo

            print("enter your move again, a2a4 FOR EXAMPLE, notice that the board is opposite. ")
            input_move = input("HUMAN, Enter your move: ")
            move = chess.Move.from_uci(input_move)
            board.push(move)


        if board.is_checkmate():
            print("GAME OVER - WHITE WON THE GAME")
            print({"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                   "ave time per move": total_time / num_of_moves / 2})

            return {"WINNER": chess.WHITE, "num_of_moves": count_moves(num_of_moves),
                    "ave time per move": total_time / num_of_moves / 2}

        start_time = time.time()

        #  TODO - wrap with a function due to repeated code in other methods
        BLACK_AI_move = ai_move(board, chess.BLACK, 3, space_M_BLACK, capture_M_BLACK, pawn_structure_M_BLACK,
                                connected_rooks_M_BLACK,
                                enemy_king_magnet_M_BLACK, the_defending_bishop_M_BLACK,
                                defending_vs_attacking_M_BLACK)
        end_time = time.time()
        total_time += end_time - start_time
        print(total_time)

        print("BLACK_AI_move", BLACK_AI_move)
        board.push(BLACK_AI_move)
        print("-------------------")
        print("BLACK AI  played", BLACK_AI_move)
        print_board(board)
        num_of_moves = num_of_moves + 1
        print("---------count_moves----------", count_moves(num_of_moves))

        if board.is_checkmate():
            print("GAME OVER - BLACK WON THE GAME")
            print(total_time)
            print({"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                   "ave time per move": total_time / num_of_moves / 2})

            return {"WINNER": chess.BLACK, "num_of_moves": num_of_moves,
                    "ave time per move": total_time / num_of_moves / 2}


'''
generates NUM_OF_GAMES_ games against stockfish15, returns the score.

@:param DNA:
        num_space_M_BLACK,
        num_capture_M_BLACK,
        num_pawn_structure_M_BLACK,
        num_connected_rooks_M_BLACK,
        num_enemy_king_magnet_M_BLACK,
        num_the_defending_bishop_M_BLACK,
        num_defending_vs_attacking_M_BLACK
        
@:param NUM_OF_GAMES_ -> how many games to play
@:return dna_score-> the DNA score against stockfish15, and the DNA
'''


def game_in_genetic_algo(
        num_space_M_BLACK,
        num_capture_M_BLACK,
        num_pawn_structure_M_BLACK,
        num_connected_rooks_M_BLACK,
        num_enemy_king_magnet_M_BLACK,
        num_the_defending_bishop_M_BLACK,
        num_defending_vs_attacking_M_BLACK):
    NUM_OF_GAMES_ = 3
    game_data_dict = {"WINNER": 0, "num_of_moves": 0, "ave time per move": 0}
    games_list = []

    for i in range(NUM_OF_GAMES_):
        game_data = game_loop(
            space_M_BLACK=num_space_M_BLACK,
            capture_M_BLACK=num_capture_M_BLACK,
            pawn_structure_M_BLACK=num_pawn_structure_M_BLACK,
            connected_rooks_M_BLACK=num_connected_rooks_M_BLACK,
            enemy_king_magnet_M_BLACK=num_enemy_king_magnet_M_BLACK,
            the_defending_bishop_M_BLACK=num_the_defending_bishop_M_BLACK,
            defending_vs_attacking_M_BLACK=num_defending_vs_attacking_M_BLACK)

        games_list.append({"num_of_moves": game_data["num_of_moves"],
                           "ave time per move": game_data["ave time per move"]})

        game_data_dict["WINNER"] += game_data["WINNER"]
        game_data_dict["num_of_moves"] += game_data["num_of_moves"]
        game_data_dict["ave time per move"] += game_data["ave time per move"]

    game_data_dict["num_of_moves"] = game_data_dict["num_of_moves"] / NUM_OF_GAMES_
    game_data_dict["ave time per move"] = game_data_dict["ave time per move"] / NUM_OF_GAMES_

    print(game_data_dict)
    print(games_list)

    dna_score = [game_data_dict, [num_space_M_BLACK,
                                  num_capture_M_BLACK,
                                  num_pawn_structure_M_BLACK,
                                  num_connected_rooks_M_BLACK,
                                  num_enemy_king_magnet_M_BLACK,
                                  num_the_defending_bishop_M_BLACK,
                                  num_defending_vs_attacking_M_BLACK]]

    return dna_score


'''
generates a random DNA 

@:param max -> the max for the random method.  
@:return random DNA
'''


def get_random_DNA(max):
    return {"num_space_M": random.choice(range(1, max)),
            "num_capture_M": random.choice(range(1, max)),
            "num_pawn_structure_M": random.choice(range(1, max)),
            "num_connected_rooks_M": random.choice(range(1, max)),
            "num_enemy_king_magnet_M": random.choice(range(1, max)),
            "num_the_defending_bishop_M": random.choice(range(1, max)),
            "num_defending_vs_attacking_M": random.choice(range(1, max))}


'''
generates a score of a random DNA against stockfish15

@:param max -> the max for the random method.  
@:return dna_score-> the DNA score against stockfish15, and the DNA

'''


def generating_random_DNA_winner(max):
    random_DNA = get_random_DNA(max)
    print("GENERATED RANDOMLY ->  ", random_DNA)

    return game_in_genetic_algo(random_DNA["num_space_M"],
                                random_DNA["num_capture_M"],
                                random_DNA["num_pawn_structure_M"],
                                random_DNA["num_connected_rooks_M"],
                                random_DNA["num_enemy_king_magnet_M"],
                                random_DNA["num_the_defending_bishop_M"],
                                random_DNA["num_defending_vs_attacking_M"])


'''
main genetic algorythm.
generates the best DNA against stockfish15. 

@:return the best DNA that the genetic algorythm has found
'''


def genetic_algo():
    max_range = random.choice(range(0, 1000))
    _DNA_1 = generating_random_DNA_winner(max_range)
    _DNA_1_sharper = sharper_dna(_DNA_1)
    print("FINAL _DNA_1 IS", _DNA_1_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_2 = generating_random_DNA_winner(max_range)
    _DNA_2_sharper = sharper_dna(_DNA_2)

    print("FINAL _DNA_2 IS", _DNA_2_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_3 = generating_random_DNA_winner(max_range)
    _DNA_3_sharper = sharper_dna(_DNA_3)

    print("FINAL _DNA_3 IS", _DNA_3_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_4 = generating_random_DNA_winner(max_range)
    _DNA_4_sharper = sharper_dna(_DNA_4)

    print("FINAL _DNA_4 IS", _DNA_4_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_5 = generating_random_DNA_winner(max_range)
    _DNA_5_sharper = sharper_dna(_DNA_5)

    print("FINAL _DNA_5 IS", _DNA_5_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_6 = generating_random_DNA_winner(max_range)
    _DNA_6_sharper = sharper_dna(_DNA_6)

    print("FINAL _DNA_6 IS", _DNA_6_sharper)

    max_range = random.choice(range(0, 1000))
    _DNA_7 = generating_random_DNA_winner(max_range)
    _DNA_7_sharper = sharper_dna(_DNA_7)

    print("FINAL _DNA_7 IS", _DNA_7_sharper)

    DNA_list = [_DNA_1_sharper, _DNA_2_sharper, _DNA_3_sharper, _DNA_4_sharper, _DNA_5_sharper, _DNA_6_sharper,
                _DNA_7_sharper]

    print("DNA_list", DNA_list)

    best_DNA = _DNA_1_sharper

    for i in range(7):

        print("DNA_list[i][0][num_of_moves]", int(DNA_list[i][0]["num_of_moves"]))
        print("best_DNA[0][num_of_moves]", int(best_DNA[0]["num_of_moves"]))

        print("DNA_list[i][0][num_of_moves]", DNA_list[i][0]["num_of_moves"])
        print("best_DNA[0][num_of_moves]", best_DNA[0]["num_of_moves"])

        if DNA_list[i][0]["num_of_moves"] >= best_DNA[0]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[1]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[2]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[3]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[4]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[5]["num_of_moves"] and \
                DNA_list[i][0]["num_of_moves"] >= best_DNA[6]["num_of_moves"]:
            return DNA_list[i]


'''
takes a DNA, and makes it 'sharper' - finds near DNA that is better. 

@:param _DNA_ 
@:return current_best-> near DNA, that is better than the current DNA. 
'''


def sharper_dna(_DNA_):
    print("MAKING THE DNA SHARPER", _DNA_)
    best_num_of_moves = _DNA_[0]["num_of_moves"]

    current_best = _DNA_

    for i in range(int(_DNA_[0]["num_of_moves"])):

        max_range0 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range1 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range2 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range3 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range4 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range5 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))
        max_range6 = random.choice(range(0, 60 - int(_DNA_[0]["num_of_moves"])))

        results = game_loop(_DNA_[1][0] + max_range0, _DNA_[1][1] + max_range1, _DNA_[1][2] + max_range2,
                            _DNA_[1][3] + max_range3, _DNA_[1][4] + max_range4, _DNA_[1][5] + max_range5,
                            _DNA_[1][6] + max_range6)

        print("best_num_of_moves", best_num_of_moves)

        if results["num_of_moves"] > best_num_of_moves:
            current_best = [_DNA_[1][0] + max_range0, _DNA_[1][1] + max_range1, _DNA_[1][2] + max_range2,
                            _DNA_[1][3] + max_range3, _DNA_[1][4] + max_range4, _DNA_[1][5] + max_range5,
                            _DNA_[1][6] + max_range6]

            best_num_of_moves = results["num_of_moves"]

    print(_DNA_, "GOT SHARPED INTO ->", current_best)
    return current_best


'''
generates a move from a theory book. 

@:param board 
@:return move-> move if exists, 0 otherwise
'''


def theory_book(board):
    try:
        """
        PLACE YOUR FILE LOCATION HERE->
        """
        move = chess.polyglot.MemoryMappedReader \
            ("computer_data.bin").weighted_choice(board).move
        print("move taken from computer book")
        return move
    except:
        return 0


'''
generates an ai move. 

@:param board 
@:param turn 
@:param depth 
@:param DNA-> space_M, capture_M, pawn_structure_M,connected_rooks_M,
              enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M
@:return move-> or via minimax algorythm, or via theory book
'''


def ai_move(board, turn, depth, space_M, capture_M, pawn_structure_M,
            connected_rooks_M,
            enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M):
    move = theory_book(board)
    # print(move)
    if move == 0:  # didn't find a play in the playbook
        return minimax(board, turn, depth, space_M, capture_M, pawn_structure_M, connected_rooks_M,
                       enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M)
    # #
    else:
        return move


'''
evaluation functions bach. 
computes the overall evaluation. 

@:param board -> given position
@:param turn -> who's turn is it
@:param evaluation functions-> space_M, capture_M, enemy_king_magnet_M, pawn_structure_M, connected_rooks_M,
                         defending_vs_attacking_M, the_defending_bishop_M
@:return total_ev-> overall evaluation of given position
'''


def evaluation_functions(board, turn, space_M, capture_M, enemy_king_magnet_M, pawn_structure_M, connected_rooks_M,
                         defending_vs_attacking_M, the_defending_bishop_M):
    # print("--------------------------------")
    total_ev = evaluation_by_pieces_location(board)
    # print(total_ev)
    total_ev = total_ev + (space_M * space(board))
    # print(total_ev)
    total_ev = total_ev + (capture_M * capture(turn, board))
    # print(total_ev)
    total_ev = total_ev + (enemy_king_magnet_M * enemy_king_magnet(turn, board))
    # print(total_ev)
    total_ev = total_ev + (pawn_structure_M * pawn_structure(turn, board))

    total_ev = total_ev + (connected_rooks_M * connected_rooks(turn, board))
    # print ("--------------")
    # print("before", total_ev)
    total_ev = total_ev + (defending_vs_attacking_M * defending_vs_attacking(turn, board))

    total_ev = total_ev + (the_defending_bishop_M * the_defending_bishop(turn, board))

    return total_ev


#####################################################

def minimax(board, turn, depth, space_M, capture_M, pawn_structure_M, connected_rooks_M,
            enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M):
    bestMove = chess.Move.null()
    bestValue = numpy.NINF - 10
    alpha = numpy.NINF
    beta = numpy.PINF
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(board, turn, -beta, -alpha, depth - 1, space_M, capture_M, pawn_structure_M,
                                connected_rooks_M,
                                enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if boardValue > alpha:
            alpha = boardValue
        board.pop()

    return bestMove


#####################################################
#####################################################

def alphabeta(board, turn, alpha, beta, depthleft, space_M, capture_M, pawn_structure_M, connected_rooks_M,
              enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M):
    bestscore = numpy.NINF

    if depthleft == 0:
        return quiesce(board, turn, alpha, beta, space_M, capture_M, pawn_structure_M, connected_rooks_M,
                       enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M)
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(board, turn, -beta, -alpha, depthleft - 1, space_M, capture_M, pawn_structure_M,
                           connected_rooks_M, enemy_king_magnet_M, the_defending_bishop_M,
                           defending_vs_attacking_M)
        board.pop()
        if score >= beta:
            return score

        if score > bestscore:
            bestscore = score

        if score > alpha:
            alpha = score
    return bestscore


#####################################################
#####################################################

def quiesce(board, turn, alpha, beta, space_M, capture_M, pawn_structure_M, connected_rooks_M,
            enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M):
    score = 0

    stand_pat = evaluation_functions(board, turn, space_M, capture_M, pawn_structure_M, connected_rooks_M,
                                     enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(board, turn, -beta, -alpha, space_M, capture_M, pawn_structure_M, connected_rooks_M,
                             enemy_king_magnet_M, the_defending_bishop_M, defending_vs_attacking_M)
            board.pop()

    if score >= beta:
        return beta

    if score > alpha:
        alpha = score

    return alpha


#####################################################
#####################################################


'''
checks how many pieces alive are there. pieces represented with self value. 
the more the better

@:param board 
@:param true_or_false -> who's turn is it
@:return total_sum-> total pieces, piece value included
'''


def left_pieces_side(board, true_or_false):
    total_sum = 0
    total_sum = total_sum + len(board.pieces(1, true_or_false))
    total_sum = total_sum + len(board.pieces(2, true_or_false)) * piece_value[2]
    total_sum = total_sum + len(board.pieces(3, true_or_false)) * piece_value[3]
    total_sum = total_sum + len(board.pieces(4, true_or_false)) * piece_value[4]
    total_sum = total_sum + len(board.pieces(5, true_or_false)) * piece_value[5]
    return total_sum


'''
checks how many pieces alive are there.
the more the better

@:param board 
'''


def left_pieces_total(board):
    return left_pieces_side(board, True) \
           + left_pieces_side(board, False)


'''
determines current game phase with respect to number of moves been played

@:param num_of_moves -> number of moves played till current status
@:return 0 , 1 , 2 (phases)
'''

# TODO - to use in more serious
def phase_generator(num_of_moves):
    if count_moves(num_of_moves) < 10:
        # opening
        return 0
    if count_moves(num_of_moves) < 25:
        # middle_game
        return 1

    else:
        # end_game
        return 2


def count_moves(num_of_moves):
    return num_of_moves


'''
evaluation function.
checks if the rooks are on the same rank without other pieces in between
connected rooks are valuable

@:param board -> current board
@:param turn -> who's turn is it
@:return count
'''


def connected_rooks(turn, board):
    count = 0
    for i in board.pieces(chess.ROOK, turn):
        for j in board.pieces(chess.ROOK, turn):
            if fixed_board[i] % 8 == fixed_board[j] % 8:
                count = count + 10
            if fixed_board[i] // 8 == fixed_board[j] // 8:
                count = count + 10

    return count


'''
evaluation function.
checks if a capture has occur
capturing enemy's pieces is valuable

@:param board -> current board
@:param turn -> who's turn is it
@:return valuation
'''


def capture(turn, board):
    move = board.pop()
    if turn == chess.BLACK:
        before = left_pieces_side(board, True)
        board.push(move)
        after = left_pieces_side(board, True)
    else:
        before = left_pieces_side(board, False)
        board.push(move)
        after = left_pieces_side(board, False)

    valuation = before - after
    return valuation


'''
evaluation function.
checks castle status relate to number of played moves. 
at the beginning castle status is valuable 

@:param num_of_moves -> moves been played
@:param board -> current board
@:param turn -> who's turn is it
@:return valuation
'''


def keep_castle(num_of_moves, turn, board):
    count = 0
    # if middle game:
    if phase_generator(num_of_moves, board) == 1:

        if turn == chess.WHITE:
            if board.piece_at(7).piece_type == KING:
                if board.piece_at(15).piece_type == PAWN:
                    count = count + 1
                if board.piece_at(16).piece_type == PAWN:
                    count = count + 1
                if board.piece_at(17).piece_type == PAWN:
                    count = count + 1

            if board.piece_at(3).piece_type == KING:
                if board.piece_at(10).piece_type == PAWN:
                    count = count + 1.5
                if board.piece_at(11).piece_type == PAWN:
                    count = count + 1.5
                if board.piece_at(9).piece_type == PAWN:
                    count = count + 1
    return count


'''
evaluation function.
haw many sqrs you are attacking. how many own sqrs are defending. 
the more the better in both categories. 
@:param board -> current board
@:param turn -> who's turn is it
@:return valuation
'''


def defending_vs_attacking(turn, board):
    attacking_1 = 0
    defending_1 = 0

    attacking_2 = 0
    defending_2 = 0

    for i in range(63):
        if board.piece_at(i):

            if board.piece_at(i).color == turn:  # black

                for square in board.attacks(i):
                    if board.piece_at(square):
                        if board.piece_at(square).color != turn:  # white:
                            attacking_1 = piece_value[board.piece_at(i).piece_type] - piece_value[
                                board.piece_at(square).piece_type]
                        if board.piece_at(square).color == turn:
                            defending_1 = piece_value[board.piece_at(i).piece_type] - piece_value[
                                board.piece_at(square).piece_type]

            if board.piece_at(i).color != turn:  # white
                for square in board.attacks(i):
                    if board.piece_at(square):
                        if board.piece_at(square).color == turn:  # black:
                            attacking_2 = piece_value[board.piece_at(i).piece_type] - piece_value[
                                board.piece_at(square).piece_type]
                        if board.piece_at(square).color != turn:
                            defending_2 = piece_value[board.piece_at(i).piece_type] - piece_value[
                                board.piece_at(square).piece_type]

    valuation = (attacking_1 - defending_2) + (attacking_2 - defending_1)
    return valuation


'''
evaluation function.
the bishop can function as an important protector of the king in some defensive setups. 
keeping the bishop in its place is valuable.

@:param board -> current board
@:param turn -> who's turn is it
@:return count-> 100 "points" upon keeping the bishop in its place. 
'''


def the_defending_bishop(turn, board):
    count = 0
    if turn == chess.WHITE:
        if board.piece_at(7):
            if board.piece_at(7).piece_type == KING:
                if board.piece_at(15):
                    if board.piece_at(15).piece_type == BISHOP and board.piece_at(15).color == turn:  # black
                        count = 100

    if turn == chess.BLACK:
        if board.piece_at(55):
            if board.piece_at(55).piece_type == BISHOP and board.piece_at(55).color != turn:
                if board.piece_at(63):
                    if board.piece_at(63).piece_type == KING:  # black
                        count = -100
    return count


'''
evaluation function.
evaluates position by proximity to the enemy's king
the closer the better

@:param board -> current board
@:param turn -> who's turn is it
@:return magnet_rank-> the sum of sqrs between all pieces to the enemy's king
'''


def enemy_king_magnet(turn, board):
    king_location = board.king(not turn)  # white
    letters_king = fixed_board[king_location] % 8
    numbers_king = fixed_board[king_location] // 8 + 1

    magnet_rank = 0

    for i in range(63):
        if board.piece_at(i):
            if board.piece_at(i).color == turn:  # black
                letters_piece = fixed_board[i] % 8
                numbers_piece = fixed_board[i] // 8 + 1
                magnet_rank = magnet_rank + (abs(letters_king - letters_piece) + abs(numbers_king - numbers_piece))

    return magnet_rank


'''
evaluation function.
evaluates position by space occupied by pieces
the more space the better.

@:param board -> current board
@:return count-> the attacked sqrs
'''


def space(board):
    i = 0
    count = 0
    while i != 64:
        count = count + len(board.attackers(False, i))
        i = i + 1

    return count


'''
evaluation function.
evaluates position by pawn structure
pawn chains are valuable. 

@:param board -> current board
@:param turn -> who's turn is it
@:return eval-> the valuation. positive/negative determined by board.turn
'''


def pawn_structure(turn, board):
    count = 0
    for i in board.pieces(chess.PAWN, turn):
        for j in board.pieces(chess.PAWN, turn):

            if abs(fixed_board[j] - fixed_board[i]) == 8:
                count = count - 4

            if fixed_board[i] % 8 == 0:
                if fixed_board[j] == fixed_board[i] + 7 or fixed_board[j] + 1 == fixed_board[i]:
                    count = count + 2

            elif fixed_board[i] % 8 == 1:
                if fixed_board[j] == fixed_board[i] + 9 or fixed_board[j] == fixed_board[i] + 1:
                    count = count + 2

            else:
                if fixed_board[j] == fixed_board[i] + 1:
                    count = count + 1

                if fixed_board[j] + 1 == fixed_board[i]:
                    count = count + 1

                if fixed_board[j] == fixed_board[i] + 9:
                    count = count + 2

                if fixed_board[j] == fixed_board[i] + 7:
                    count = count + 2
    return count


'''
evaluation function.
evaluates position by pieces location

@:param board -> current board
@:return eval-> the valuation. positive/negative determined by board.turn
'''


def evaluation_by_pieces_location(board):
    if board.is_checkmate():
        if board.turn:
            return numpy.NINF
        else:
            return numpy.PINF
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 1000 * (wp - bp) + 3200 * (wn - bn) + 3300 * (wb - bb) + 5000 * (wr - br) + 9000 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])

    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    if board.turn:
        return eval
    else:
        return -eval


'''
prints board for a visual text based board
'''


def print_board(board):
    to_print = ""
    for i in range(64):
        if board.piece_at(i):
            to_print = to_print + UNICODE_PIECE_SYMBOLS[board.piece_at(i).symbol()] + " "

        else:
            to_print = to_print + "💢 "
        if (i + 1) % 8 == 0:
            print(to_print)
            to_print = ""


if __name__ == '__main__':

    first_input = input("HELLO, AND WELCOME TO AI-CHESS BY ITAMAR, "
                        "PLEASE TYPE ONE OF THE OPTIONS AS DESCRIBED IN THE INSTRUCTIONS ABOVE FILE: ")
    print(first_input)
    if first_input == '1':
        #  human VS RANDOM AI

        random_DNA = get_random_DNA(1000)
        print("GENERATED RANDOMLY ->  ", random_DNA)

        game_loop_human(random_DNA["num_space_M"],
                        random_DNA["num_capture_M"],
                        random_DNA["num_pawn_structure_M"],
                        random_DNA["num_connected_rooks_M"],
                        random_DNA["num_enemy_king_magnet_M"],
                        random_DNA["num_the_defending_bishop_M"],
                        random_DNA["num_defending_vs_attacking_M"])

    if first_input == '2':
        random_DNA = get_random_DNA(1000)
        print("GENERATED RANDOMLY ->  ", random_DNA)
        game_loop(random_DNA["num_space_M"],
                  random_DNA["num_capture_M"],
                  random_DNA["num_pawn_structure_M"],
                  random_DNA["num_connected_rooks_M"],
                  random_DNA["num_enemy_king_magnet_M"],
                  random_DNA["num_the_defending_bishop_M"],
                  random_DNA["num_defending_vs_attacking_M"])

    else:
        print("run program again, you didn't choose a proper number my dear friend")

# game_loop()
