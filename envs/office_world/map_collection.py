from .game_objects import *

class MapCollection:
    # MAP_1_OBJECTS: dict[tuple[int, int], OfficeWorldObjects] = {
    #     (1, 1): OfficeWorldObjects.A,
    #     (1, 7): OfficeWorldObjects.B,
    #     (10, 7): OfficeWorldObjects.C,
    #     (10, 1): OfficeWorldObjects.D,
    #     (7, 4): OfficeWorldObjects.MAIL,
    #     (8, 2): OfficeWorldObjects.COFFEE, #(10, 0): OfficeWorldObjects.COFFEE
    #     (3, 6): OfficeWorldObjects.COFFEE,
    #     (4, 4): OfficeWorldObjects.OFFICE,
    #     (4, 1): OfficeWorldObjects.PLANT,
    #     (7, 1): OfficeWorldObjects.PLANT,
    #     (4, 7): OfficeWorldObjects.PLANT,
    #     (7, 7): OfficeWorldObjects.PLANT,
    #     (1, 4): OfficeWorldObjects.PLANT,
    #     (10, 4): OfficeWorldObjects.PLANT,
    # }
    
    # MAP_2_OBJECTS: dict[tuple[int, int], str] = {
    #     (7, 4): OfficeWorldObjects.MAIL,
    #     (10, 0): OfficeWorldObjects.COFFEE,
    #     (10, 8): OfficeWorldObjects.COFFEE,
    #     (4, 4): OfficeWorldObjects.OFFICE,
    #     (4, 1): OfficeWorldObjects.PLANT,
    #     (7, 1): OfficeWorldObjects.PLANT,
    #     (4, 7): OfficeWorldObjects.PLANT,
    #     (7, 7): OfficeWorldObjects.PLANT,
    #     (1, 4): OfficeWorldObjects.PLANT,
    #     (10, 4): OfficeWorldObjects.PLANT,
    # }

    # MAP_3_OBJECTS: dict[tuple[int, int], str] = {
    #     (7, 4): OfficeWorldObjects.MAIL,
    #     (10, 0): OfficeWorldObjects.COFFEE,
    #     (3, 6): OfficeWorldObjects.COFFEE,
    #     (4, 4): OfficeWorldObjects.OFFICE,
    # }

    MAP_4_OBJECTS: dict[tuple[int, int], str] = {
        (5, 2): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (6, 3): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (1, 1): OfficeWorldObjects.OFFICE,
        (1, 4): OfficeWorldObjects.PLANT,
        (4, 4): OfficeWorldObjects.PLANT,
        (7, 4): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
    }

    MAP_5_OBJECTS: dict[tuple[int, int], str] = {
        (5, 2): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (6, 3): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (1, 1): OfficeWorldObjects.OFFICE,
    }
    
    # less decorations
    MAP_6_OBJECTS: dict[tuple[int, int], str] = {
        (5, 2): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (6, 3): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (1, 1): OfficeWorldObjects.OFFICE,
        (4, 4): OfficeWorldObjects.PLANT,
        (7, 4): OfficeWorldObjects.PLANT,
    }

    MAP_7_OBJECTS: dict[tuple[int, int], str] = {
        (3, 0): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (5, 0): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (4, 4): OfficeWorldObjects.OFFICE,
        (1, 4): OfficeWorldObjects.PLANT,
        (1, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
    }


    MAP_8_OBJECTS: dict[tuple[int, int], str] = {
        (3, 0): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (5, 0): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (4, 4): OfficeWorldObjects.OFFICE,
        (1, 4): OfficeWorldObjects.PLANT,
        # (1, 1): OfficeWorldObjects.PLANT, suiside prevention program
        (7, 4): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
    }


    MAP_9_OBJECTS: dict[tuple[int, int], str] = {
        (3, 2): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (5, 0): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (4, 4): OfficeWorldObjects.OFFICE,
        (1, 4): OfficeWorldObjects.PLANT,
        (1, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
    }

    MAP_10_OBJECTS: dict[tuple[int, int], str] = {
        (3, 0): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (5, 0): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (4, 4): OfficeWorldObjects.OFFICE,
    }

    MAP_11_OBJECTS: dict[tuple[int, int], str] = {
        (3, 0): OfficeWorldObjects.COFFEE_1, # add second coffee sign
        (5, 0): OfficeWorldObjects.COFFEE_2, # add second coffee sign
        (4, 4): OfficeWorldObjects.OFFICE,

        # (1, 1): OfficeWorldObjects.PLANT,
        # (1, 4): OfficeWorldObjects.PLANT,
        (4, 4): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (7, 4): OfficeWorldObjects.PLANT,

    }