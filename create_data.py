import math
import os
import random

import cv2
import np as np
import numpy as np
from sympy import symbols, solve

HORIZONTAL = 1
VERTICAL = 0
IMG_SIZE = 800

'''
creat the 64 squares

@:param img -> chess board image
@:param raws ->  an array of arrays. each inner array represents a row of points
@:return squares -> a dictionary of 64 squares, each represented by 4 P(x,y)
'''


def creat_squares(img, rows):
    squares = {}
    square_num = 1
    for i in range(8):
        for j in range(8):
            new_square = [[int(rows[i + 1][j][0]), int(rows[i + 1][j][1])],
                          [int(rows[i][j][0]), int(rows[i][j][1])],
                          [int(rows[i][j + 1][0]), int(rows[i][j + 1][1])],
                          [int(rows[i + 1][j + 1][0]), int(rows[i + 1][j + 1][1])]]

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_sqr(img, new_square, color)
            perspective_sqr = perspective(img, new_square)
            # show(perspective_sqr)
            resized_perspective = resize_img(perspective_sqr, 100)
            print(resized_perspective.shape)
            show(resized_perspective)

            squares[square_num] = new_square
            print("square_num", square_num)

            manual_sqr_classifier(resized_perspective)

            square_num += 1
            show(img)

    draw_move(img, squares[20], squares[10])
    show(img)
    print(squares)
    return squares


'''

classifies sqrs into different folders, getting the dataset ready. 
@:param resized_perspective -> a sqr ready to be sent to dataset

for manual classification-
BLACK 1
WHITE 0
EMPTY 0 || PAWN 1 || BISHOP 2 || KNIGHT 3 || ROOK 5 || QUEEN 9 || KING 7

'''


def manual_sqr_classifier(resized_perspective):
    b_or_w = cv2.waitKey(0)
    print('b_or_w', b_or_w)
    piece_or_sqr = cv2.waitKey(0)
    print('piece_or_sqr', piece_or_sqr)

    random_name = random.randint(0, 1000000)

    train_or_test = random.randint(0, 10)
    if train_or_test >= 7:
        train_or_test = 'train'
    else:
        train_or_test = 'test'

    #  TODO - insert repeated path to variable

    if piece_or_sqr == 49:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}p'
        print(f"{b_or_w}PAWN")
    elif piece_or_sqr == 50:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}b'
        print(f"{b_or_w}BISHOP")
    elif piece_or_sqr == 51:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}kn'
        print(f"{b_or_w}KNIGHT")
    elif piece_or_sqr == 53:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}r'
        print(f"{b_or_w}ROOK")
    elif piece_or_sqr == 57:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}q'
        print(f"{b_or_w}QUEEN")
    elif piece_or_sqr == 54:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\{b_or_w}k'
        print(f"{b_or_w}KING")
    else:
        path = f'C:\\Users\\itama\\PycharmProjects\\yael_checker\\{train_or_test}\\empty'
        print("EMPTY")
    cv2.imwrite(os.path.join(path, f'{random_name}.jpg'), resized_perspective)


'''
highlights a chosen square

@:param img -> chess board image
@:param sqr -> coordination of square in image
@:param color - > color to highlight square
@:return img -> chess board with the highlighted square drawn on it
'''


def draw_sqr(img, sqr, color):
    nor_sqr = np.array([sqr], np.int32)
    cv2.polylines(img, [nor_sqr], True, color, thickness=7)
    return img


'''
draws a move, represented by highlighting the origin(in RED) and destination(in GREEN) 
squares, while drawing an arrow in between

@:param img -> chess board image
@:param origin_sqr -> coordination of the origin square in image
@:param destination_sqr -> coordination of the destination square in image
@:return img -> chess board image with the move drawn on it
'''


def draw_move(img, origin_sqr, destination_sqr):
    draw_arrow(img, origin_sqr, destination_sqr)
    draw_sqr(img, origin_sqr, (0, 0, 255))
    draw_sqr(img, destination_sqr, (0, 255, 0))
    show(img)


'''
draws an arrow between 2 squares

@:param img -> chess board image
@:param origin_sqr -> coordination of the origin square in image
@:param destination_sqr -> coordination of the destination square in image
@:return img -> chess board image with the arrow drawn on it
'''


def draw_arrow(img, origin_sqr, destination_sqr):
    origin_sqr_X = (origin_sqr[0][0] +
                    origin_sqr[1][0] +
                    origin_sqr[2][0] +
                    origin_sqr[3][0]) / 4

    origin_sqr_Y = (origin_sqr[0][1] +
                    origin_sqr[1][1] +
                    origin_sqr[2][1] +
                    origin_sqr[3][1]) / 4

    destination_sqr_X = (destination_sqr[0][0] +
                         destination_sqr[1][0] +
                         destination_sqr[2][0] +
                         destination_sqr[3][0]) / 4

    destination_sqr_Y = (destination_sqr[0][1] +
                         destination_sqr[1][1] +
                         destination_sqr[2][1] +
                         destination_sqr[3][1]) / 4

    img_w_arrow = cv2.arrowedLine(img,
                                  (int(origin_sqr_X), int(origin_sqr_Y)),
                                  (int(destination_sqr_X), int(destination_sqr_Y)),
                                  (0, 0, 255), 5)

    show(img_w_arrow)
    return img_w_arrow


'''
converts 4 P(x,y) into an Equilateral square

@:param img -> chess board image
@:param square -> represented by 4 P(x,y)
@:return equilateral_square -> equilateral square of the given square
'''


def perspective(img, square):
    pt_A = square[0]
    pt_B = square[1]
    pt_C = square[2]
    pt_D = square[3]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    equilateral_square = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return equilateral_square


'''
sorts the points of the line intersections, 
ordered by Y coordinates followed by X coordinates. 
0.....9
.......
72...81
@:param points -> points if line intersections 
@:return rows -> an array of arrays. each inner array represents a row of points
'''


def sort_points(points):
    sorter = lambda x: (x[1], x[0])
    sorted_l = sorted(points, key=sorter)
    print(sorted_l)

    one = sorted_l[0:9]
    two = sorted_l[9:18]
    three = sorted_l[18:27]
    four = sorted_l[27:36]
    five = sorted_l[36:45]
    six = sorted_l[45:54]
    seven = sorted_l[54:63]
    eight = sorted_l[63:72]
    nine = sorted_l[72:81]

    rows = [one, two, three, four, five, six, seven, eight, nine]

    return rows


'''
opens a window, showing the given image
@:param img -> the image to show
'''


def show(img):
    cv2.imshow(str(img), img)
    cv2.waitKey(0)


'''
resizes the given image, with respect to the given scale
@:param img -> the image to show
@:param scale -> scale
'''


def resize_img(img, scale):
    resized = cv2.resize(img, (int(scale), int(scale)))
    return resized


'''
convert the image into a grayed-blurred state

@:param img -> an image, loaded using cv2.imread()
@:return gray_blur -> the image at a grayed-blurred state
'''


def gray_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    return gray_blur


'''
reads an image

@:param file -> the 'path' of the image
@:return img -> the image, ready to be shown
'''


# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file), 1)
    return img


'''
finds edges in a given image using  cv2.Canny

@:param img -> an image, loaded using cv2.imread()
@:return edged -> the edges on image, calculated using cv2.Canny()
'''


def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    # c_edges = cv2.Canny(img, 190, 200)
    return edged


'''
finds straight lines in a given cv2.Canny() output (edges in original image).
draws the lines on original given image

@:param img -> an image, loaded using cv2.imread()
@:param edges -> the edges on image, calculated using cv2.Canny()

@:return h_lines -> horizontal lines
@:return v_lines -> vertical lines
@:return img -> the original given image, with the drawn lines on it
'''


def hough_line(edges, img):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    h_lines = {}
    v_lines = {}

    cartesian_lines_list = polar_to_cartesian(lines)
    for cartesian_line in cartesian_lines_list:

        x1 = cartesian_line[0]
        y1 = cartesian_line[1]
        x2 = cartesian_line[2]
        y2 = cartesian_line[3]
        r = cartesian_line[4]
        theta = cartesian_line[5]

        print("x1", x1, "////////", "y1", y1, "/////", "x2", x2, "//////", "y2", y2)
        exists = False
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            print("v")
            #  TODO - replace code with 'is_close_line_exists' function

            for line in v_lines.values():
                if math.isclose(line[0][0], x1, abs_tol=50) or \
                        math.isclose(line[1][0], x2, abs_tol=50):
                    exists = True

            if not exists:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                show(img)
                v_lines[(r, theta)] = [(x1, y1), (x2, y2)]
        else:
            print("h")
            for line in h_lines.values():
                #  TODO - replace code with 'is_close_line_exists' function

                if math.isclose(line[0][1], y1, abs_tol=50) or \
                        math.isclose(line[1][1], y2, abs_tol=50):
                    exists = True

            if not exists:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 80), 2)
                show(img)
                # h_lines.append((r, theta))
                h_lines[(r, theta)] = [(x1, y1), (x2, y2)]

    cv2.imwrite('linesDetected.jpg', img)

    return h_lines, v_lines, img


'''
changes the lines from a polar representation into a cartesian representation

@:param lines -> lines -> a cv2.HoughLines output lines. polar representation (r, theta)

@:return cartesian_lines_list -> the given lines, cartesian representation ( r, theta are saved)
line = [x1, y1, x2, y2, r, theta]
'''


def polar_to_cartesian(lines):
    cartesian_lines_list = []
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
        # x0 stores the value rcos(theta)
        x0 = a * r
        # y0 stores the value rsin(theta)
        y0 = b * r
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + IMG_SIZE * (-b))
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + IMG_SIZE * (a))
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - IMG_SIZE * (-b))
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - IMG_SIZE * (a))

        new_car_line = [x1, y1, x2, y2, r, theta]
        cartesian_lines_list.append(new_car_line)

    return cartesian_lines_list


'''
checks if there is already an exists close line. 
( cv2.HoughLines might detect a line as a few close lines. 
the method "converts" the bach of lines into a single line)

@:param v_or_h_lines -> list of existing lines
@:param x1, y1, x2, y2, r, theta -> new line. 

@:return h_lines -> horizontal lines
@:return v_lines -> vertical lines
@:return img -> the original given image, with the drawn lines on it
'''


def is_close_line_exists(v_or_h_lines, x1, y1, x2, y2, r, theta):
    exists = False
    for line in v_or_h_lines.values():
        if math.isclose(line[0][1], y1, abs_tol=50) or \
                math.isclose(line[1][1], y2, abs_tol=50):
            print("exists ", "y1", line[0][1], "y2", line[1][1])
            exists = True

    if not exists:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 80), 2)
        show(img)
        # h_lines.append((r, theta))
        v_or_h_lines[(r, theta)] = [(x1, y1), (x2, y2)]

    return v_or_h_lines


'''
add a bach of new lines to old lines

@:param old_lines -> bach of new lines represented by 
#  [{'new_x1': 226, 'new_y1': 801, 'new_x2': 239, 'new_y2': -798},

@:param new_lines -> bach of existing-old, lines represented by 
#  [[(-800, 24), (799, 51)], [(-800, 665),

@:return old_lines -> old lines with new line all together
'''


def add_lines(old_lines, new_lines):
    for i, new_line in enumerate(new_lines):
        new = [1, 1]
        new[0] = (new_line['new_x1'], new_line['new_y1'])
        new[1] = new_line['new_x2'], new_line['new_y2']
        old_lines[i] = new

    return old_lines


'''
finds straight lines in a given cv2.Canny() output (edges in original image).
draws the lines on original given image

@:param h_lines -> HORIZONTAL lines
@:param v_lines -> VERTICAL lines
@:param img -> given image

@:return points -> list of points of intersection between the HORIZONTAL and VERTICAL lines
'''


def line_intersections(h_lines, v_lines, img):
    h_lines_linear_eq = []
    v_lines_linear_eq = []
    points = []

    for line in h_lines:
        new_line_linear_eq = from_xy_to_linear_eq(line)
        h_lines_linear_eq.append(new_line_linear_eq)

    for line in v_lines:
        new_line_linear_eq = from_xy_to_linear_eq(line)

        v_lines_linear_eq.append(new_line_linear_eq)

    for h_line in h_lines_linear_eq:
        for v_line in v_lines_linear_eq:
            x, y = point_inter(h_line, v_line)

            if (x and y) != np.Infinity:
                points.append([x, y])
                cv2.circle(img, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
                show(img)
                print('point!!', int(x), int(y))
    return points


'''
finds point of intersection between two given lines

@:param line1 -> line 1
@:param line2 -> line 2

@:return point_x -> x of point
@:return point_y -> y of point

returns --np.Infinity, np.Infinity-- only if the point of intersection is
out of the scope fo the IMG_SIZE
'''


def point_inter(line1, line2):
    m1 = line1[2]
    n1 = line1[3]

    m2 = line2[2]
    n2 = line2[3]

    x = symbols('x')
    expr = (m1 * x) + n1 - (m2 * x) - n2

    try:
        point_x = int(solve(expr)[0])
        point_y = int(solve(expr)[0] * m1 + n1)

        if (point_x <= IMG_SIZE) or (point_y <= IMG_SIZE):
            print('POINT AT-->', point_x, point_y)
            return point_x, point_y
        else:
            print("NO POINT, TOO BIG")
            print('POINT AT-->', point_x, point_y)


            return np.Infinity, np.Infinity

    except:
        print("NO POINT out of range")
        return np.Infinity, np.Infinity


'''
converts a line, represented by 2 (x,y) points, to a linear equation

@:param line -> a line, represented by 2 (x,y) points
@:return h_or_v_lines_cartes -> the given line, represented as a linear equation [ x, y, m, n ]
'''


def from_xy_to_linear_eq(line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    if x1 == x2:
        m = (y2 - y1) / (x2 - x1 + 1)
    else:
        m = (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    h_or_v_lines_cartes = [int(x1), int(y1), int(m), int(n)]
    return h_or_v_lines_cartes



'''
collects gaps lengths using  find_specified_gaps() 
determent the side's length of the common square

@:param h_lines -> horizontal lines
@:param v_lines -> vertical lines

@:return final_square_length -> the side's length of an average square
@:return all_gaps -> a dictionary that holds the gaps between the edges of the line 
  e.g -> all_gaps['h_lines_left'] : [ 23, 43, 43] ...
'''


def determine_square_length(h_lines, v_lines):
    # v
    top = [0, 0]
    down = [1, 0]

    # h
    right = [0, 1]
    left = [1, 1]

    gaps_sum = {}
    all_gaps = {}
    v_top, v_top_amount, all_gaps['v_lines_top'] = find_gaps(v_lines, top)
    if v_top in gaps_sum.keys():
        gaps_sum[v_top] += v_top_amount
    else:
        gaps_sum[v_top] = v_top_amount

    v_down, v_down_amount, all_gaps['v_lines_down'] = find_gaps(v_lines, down)
    if v_down in gaps_sum.keys():
        gaps_sum[v_down] += v_down_amount
    else:
        gaps_sum[v_down] = v_down_amount

    h_right, h_right_amount, all_gaps['h_lines_right'] = find_gaps(h_lines, right)
    if h_right in gaps_sum.keys():
        gaps_sum[h_right] += h_right_amount
    else:
        gaps_sum[h_right] = h_right_amount

    h_left, h_left_amount, all_gaps['h_lines_left'] = find_gaps(h_lines, left)
    if h_left in gaps_sum.keys():
        gaps_sum[h_left] += h_left_amount
    else:
        gaps_sum[h_left] = h_left_amount

    final_square_length = max(gaps_sum, key=gaps_sum.get)

    return final_square_length, all_gaps


'''
collects gaps lengths using  find_specified_gaps() 
determines the side's length of an average square

@:param v_or_h_lines -> horizontal lines or vertical lines
@:param gap_category -> right or left, up or down

@:return current_best_gap -> the most common gap
@:return current_max -> how many times appeared
@:return gaps_list -> a dictionary that holds the gaps between the [gap_category] edges of the lines [v_or_h_lines]
  e.g -> all_gaps['h_lines_left'] : [ 23, 43, 43] ...
'''
#  TODO to add a similar method that finds missing line on the edge of the image.


def find_gaps(v_or_h_lines, gap_category):
    edge = gap_category[0]
    x_or_y = gap_category[1]

    sorter = lambda x: (x[edge][x_or_y])
    v_or_h_lines_sorted = sorted(v_or_h_lines, key=sorter)
    print(v_or_h_lines_sorted)

    gaps_list = []
    gaps_groups = {}
    for i, x in enumerate(v_or_h_lines_sorted):

        if i + 1 < len(v_or_h_lines_sorted):  # just to enable the check.
            new_gap = abs(x[edge][x_or_y] - v_or_h_lines_sorted[i + 1][edge][x_or_y])
            gaps_list.append(new_gap)
            gap_group_exists = False
            for gap_range in gaps_groups.keys():

                abs_tol = gap_range / 10
                if math.isclose(gap_range, new_gap, abs_tol=abs(abs_tol)):
                    gap_group_exists = True
                    gaps_groups[gap_range].append(new_gap)

            if not gap_group_exists:
                gaps_groups[new_gap] = [new_gap]

    current_max = 0
    current_best_gap = 0
    for gap_range in gaps_groups.values():
        if len(gap_range) > current_max:
            current_best_gap = np.mean(gap_range)
            current_max = len(gap_range)

    return current_best_gap, current_max, gaps_list


'''
finds and draws the missing lines accused by line gaps.

@:param img -> an image, loaded using cv2.imread()
@:param gaps -> dictionary of line gaps
@:param square_length -> dictionary of line gaps
@:param h_or_v_lines -> dictionary of line gaps
@:param h_or_v -> VERTICAL or HORIZONTAL

@:return founded_lines -> founded missing lines
'''


def find_missing_line_gaps(img, gaps, square_length, h_or_v_lines, h_or_v):
    sorter = lambda x: (x[0][h_or_v])
    lines_sorted = sorted(h_or_v_lines.values(), key=sorter)

    if h_or_v == VERTICAL:
        str_h_or_r = 'v_lines_top'
    else:
        str_h_or_r = 'h_lines_right'

    new_lines = []
    founded_lines = []
    for i, gap in enumerate(gaps[str_h_or_r]):
        if not math.isclose(gap, square_length, abs_tol=square_length / 2):

            missing_squares = int(gap / square_length)
            print('missing_squares', missing_squares)
            print(f"{i} is problematic{h_or_v}")
            print('gap', gap, 'square_length', square_length, 'square_length/3', square_length / 3)

            for j in range(missing_squares):
                f_x1 = lines_sorted[i][0][0]
                f_y1 = lines_sorted[i][0][1]
                f_x2 = lines_sorted[i][1][0]
                f_y2 = lines_sorted[i][1][1]

                s_x1 = lines_sorted[i + 1][0][0]
                s_y1 = lines_sorted[i + 1][0][1]
                s_x2 = lines_sorted[i + 1][1][0]
                s_y2 = lines_sorted[i + 1][1][1]

                new_lines = find_lines_location_in_gaps(h_or_v, gap, square_length, f_x1, f_y1, f_x2, f_y2,
                                                        abs(s_x1), s_y1, s_x2, s_y2)

            for line in new_lines:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.line(img, (line['new_x1'], line['new_y1']), (line['new_x2'], line['new_y2']), color, 5)
                founded_lines.append(line)

    return founded_lines


'''
finds desired lines location in a gap

@:param h_or_v -> VERTICAL or HORIZONTAL
@:param gap -> a gap's length
@:param square_length -> dictionary of line gaps

@:param f_x1 -> first line, first edge, X coordinate
@:param f_y1 -> first line, first edge, Y coordinate
@:param f_x2 -> first line, second edge, X coordinate
@:param f_y2 -> first line, second edge, Y coordinate

@:param s_x1 -> second line, first edge, X coordinate
@:param s_y1 -> second line, first edge, Y coordinate
@:param s_x2 -> second line, second edge, X coordinate
@:param s_y2 -> second line, second edge, Y coordinate


@:return h_lines -> horizontal lines
@:return v_lines -> vertical lines
@:return img -> the original given image, with the drawn lines on it
'''


def find_lines_location_in_gaps(h_or_v, gap, square_length, f_x1, f_y1, f_x2, f_y2, s_x1, s_y1, s_x2, s_y2):
    inner_squares = round(gap / square_length)
    lines_locations = []
    print('inner_squares', inner_squares)
    for i in range(inner_squares - 1):

        if h_or_v == HORIZONTAL:

            new_x1 = f_x1
            new_x2 = s_x1
            new_y1 = (f_y1 + s_y1) * ((i + 1) / inner_squares)
            new_y2 = (f_y2 + s_y2) * ((i + 1) / inner_squares)

        else:

            new_y1 = np.mean([f_y1, s_y1])
            new_y2 = np.mean([f_y2, s_y2])
            if new_y1 < 0 and new_y2 < 0:
                new_y2 = IMG_SIZE

            new_x1 = f_x1 + (s_x1 - f_x1) * ((i + 1) / inner_squares)
            new_x2 = f_x2 + (s_x2 - f_x2) * ((i + 1) / inner_squares)

        new_line = {'new_x1': int(new_x1), 'new_y1': int(new_y1), 'new_x2': int(new_x2), 'new_y2': int(new_y2)}

        lines_locations.append(new_line)

    return lines_locations



img = read_img('SHAY_2.jpeg')
resized = resize_img(img, IMG_SIZE)
cv2.circle(resized, (int(-200), int(-200)), radius=50, color=(255, 0, 0), thickness=-1)

show(resized)

print(resized.shape)
gray_blur = gray_blur(resized)
edges = canny_edge(gray_blur)

h_lines, v_lines, img2 = hough_line(edges, resized)

sorter_v = lambda x: (x[0][VERTICAL])
v_lines_sorted = sorted(v_lines.values(), key=sorter_v)

sorter_h = lambda x: (x[0][HORIZONTAL])
h_lines_sorted = sorted(h_lines.values(), key=sorter_h)

square_length, gaps = determine_square_length(h_lines_sorted, v_lines_sorted)
print('square_length, gaps-->', square_length, gaps)
print(v_lines.values())

v_added_lines = find_missing_line_gaps(img2, gaps, square_length, v_lines, VERTICAL)
h_added_lines = find_missing_line_gaps(img2, gaps, square_length, h_lines, HORIZONTAL)

print('v_added_lines', len(v_added_lines), '--', v_added_lines)

v_lines = add_lines(v_lines, v_added_lines)
h_lines = add_lines(h_lines, h_added_lines)

print("BBBBBBBBBBB", len(v_lines))
print("BBBBBBBBBBB", len(h_lines))

if (len(v_lines) == 9) and (len(h_lines) == 9):
    # points = line_intersections(h_lines.keys(), v_lines.keys(), img2)
    points = line_intersections(h_lines.values(), v_lines.values(), img2)

    show(img)
    rows = sort_points(points)
    squars = creat_squares(resized, rows)
    show(img)
else:
    print('NOT A GOOD PICTURE, COULDNT IDENTIFY 9 LINES X2')

# crop the image a little bit diffrently, for example - higher, righter and so.
# OBJECT DIDECTION OF THE BOARD ITSELF TO STIFF THE BOUNDRIES USING ->
# https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images DATASET

