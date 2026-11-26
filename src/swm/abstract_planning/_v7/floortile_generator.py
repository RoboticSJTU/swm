# Author: Tomas de la Rosa
# MIT license in _v7/assets/pddl_generators/floortile/license
#
import random


def get_tile(row, col):
    return "tile_" + str(row) + "-" + str(col)


# There is an extra row below to place robots at the end


def get_objects(num_rows, num_columns, num_robots):
    #   TILES
    str_tiles = ""
    for row in range(num_rows + 1):
        for column in range(num_columns):
            str_tiles = str_tiles + get_tile(row, column + 1) + " "
        if row != num_rows:
            str_tiles += "\n           "
    str_tiles += "- tile\n"

    # ROBOTS
    str_robots = "           "
    for robot in range(num_robots):
        str_robots += "robot" + str(robot + 1) + " "
    str_robots += "- robot\n"

    # COLORS
    str_colors = "           white black - color\n"
    return str_tiles + str_robots + str_colors


def get_init(num_rows, num_columns, num_robots, mode_flag, rng):
    robot_locations = []
    str_init = "\n"
    if mode_flag != "time":
        str_init += "   (= (total-cost) 0)\n"

    used_cols = list(range(1, num_columns + 1))

    for robot in range(num_robots):
        posx = rng.randint(0, num_rows)
        posy = rng.choice(used_cols)
        robot_pos = get_tile(posx, posy)
        robot_locations += [robot_pos]
        used_cols.remove(posy)

        str_init += "   (robot-at robot" + str(robot + 1) + " " + robot_pos + ")\n"
        if robot % 2 == 0:
            robot_color = "white"
        else:
            robot_color = "black"
        str_init += "   (robot-has robot" + str(robot + 1) + " " + robot_color + ")\n"

    str_init += "   (available-color white)\n"
    str_init += "   (available-color black)\n"

    for row in range(num_rows + 1):
        for column in range(num_columns):
            if get_tile(row, column + 1) not in robot_locations:
                str_init += "   (clear " + get_tile(row, column + 1) + ")\n"

    for row in range(num_rows):
        for column in range(num_columns):
            str_init += (
                "   (up "
                + get_tile(row + 1, column + 1)
                + " "
                + get_tile(row, column + 1)
                + ")\n"
            )
    for row in range(num_rows):
        for column in range(num_columns):
            str_init += (
                "   (down "
                + get_tile(row, column + 1)
                + " "
                + get_tile(row + 1, column + 1)
                + ")\n"
            )
    for row in range(num_rows + 1):
        for column in range(num_columns - 1):
            str_init += (
                "   (right "
                + get_tile(row, column + 2)
                + " "
                + get_tile(row, column + 1)
                + ")\n"
            )
    for row in range(num_rows + 1):
        for column in range(num_columns - 1):
            str_init += (
                "   (left "
                + get_tile(row, column + 1)
                + " "
                + get_tile(row, column + 2)
                + ")\n"
            )

    return str_init


def get_goals(num_rows, num_columns):
    str_goals = "(and\n"
    for row in range(num_rows):
        for column in range(num_columns):
            if (row + column) % 2 == 0:
                str_goals += (
                    "    (painted " + get_tile(row + 1, column + 1) + " white)\n"
                )
            else:
                str_goals += (
                    "    (painted " + get_tile(row + 1, column + 1) + " black)\n"
                )
    return str_goals + ")"


def generate(name, num_rows, num_columns, num_robots, mode_flag, seed):
    rng = random.Random(seed)
    objects = get_objects(num_rows, num_columns, num_robots)
    initial = get_init(num_rows, num_columns, num_robots, mode_flag, rng)
    goal = get_goals(num_rows, num_columns)
    metric = "total-time" if mode_flag == "time" else "total-cost"
    return (f"(define (problem {name})\n (:domain floor-tile)\n (:objects {objects})\n"
            f" (:init {initial})\n (:goal {goal})\n (:metric minimize ({metric}))\n)\n")
