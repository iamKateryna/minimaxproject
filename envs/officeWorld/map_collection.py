from .game_objects import *

class MapCollection:
    MAP_1_OBJECTS: dict[tuple[int, int], OfficeWorldObjects] = {
        (1, 1): OfficeWorldObjects.A,
        (1, 7): OfficeWorldObjects.B,
        (10, 7): OfficeWorldObjects.C,
        (10, 1): OfficeWorldObjects.D,
        (7, 4): OfficeWorldObjects.MAIL,
        (8, 2): OfficeWorldObjects.COFFEE, #(10, 0): OfficeWorldObjects.COFFEE
        (3, 6): OfficeWorldObjects.COFFEE,
        (4, 4): OfficeWorldObjects.OFFICE,
        (4, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (4, 7): OfficeWorldObjects.PLANT,
        (7, 7): OfficeWorldObjects.PLANT,
        (1, 4): OfficeWorldObjects.PLANT,
        (10, 4): OfficeWorldObjects.PLANT,
    }
    
    MAP_2_OBJECTS: dict[tuple[int, int], str] = {
        (7, 4): OfficeWorldObjects.MAIL,
        (10, 0): OfficeWorldObjects.COFFEE,
        (10, 8): OfficeWorldObjects.COFFEE,
        (4, 4): OfficeWorldObjects.OFFICE,
        (4, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (4, 7): OfficeWorldObjects.PLANT,
        (7, 7): OfficeWorldObjects.PLANT,
        (1, 4): OfficeWorldObjects.PLANT,
        (10, 4): OfficeWorldObjects.PLANT,
    }

    MAP_3_OBJECTS: dict[tuple[int, int], str] = {
        (7, 4): OfficeWorldObjects.MAIL,
        (10, 0): OfficeWorldObjects.COFFEE,
        (3, 6): OfficeWorldObjects.COFFEE,
        (4, 4): OfficeWorldObjects.OFFICE,
    }

    MAP_4_OBJECTS: dict[tuple[int, int], str] = {
        (5, 2): OfficeWorldObjects.COFFEE,
        (6, 3): OfficeWorldObjects.COFFEE,
        (1, 1): OfficeWorldObjects.OFFICE,
        (1, 4): OfficeWorldObjects.PLANT,
        (4, 4): OfficeWorldObjects.PLANT,
        (7, 4): OfficeWorldObjects.PLANT,
        (4, 1): OfficeWorldObjects.PLANT,
    }