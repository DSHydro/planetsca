import json

import fiona
from shapely.geometry import mapping, shape
from shapely import concave_hull, unary_union
from shapely.geometry import Polygon, mapping



def get_coordinates(file_path: str) -> list:
    """
    Helper method for extracting a list of coordinates from a GeoJSON file

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - list: Coordinates of a given GeoJSON file
    """
    with open(file_path) as f:
        data = json.load(f)

    coordinates_list = []

    for feature in data["features"]:
        geometry = feature["geometry"]
        geometry_type = geometry["type"]
        coordinates = geometry["coordinates"]

        if geometry_type in ["Point", "LineString"]:
            coordinates_list.append(coordinates)
        elif geometry_type == "Polygon":
            for polygon in coordinates:
                coordinates_list.extend(polygon)
        elif geometry_type == "MultiPolygon":
            for multipolygon in coordinates:
                for polygon in multipolygon:
                    coordinates_list.extend(polygon)
    return coordinates_list


def vertex_count(file_path: str) -> int:
    """
    Counts vertexes from a GeoJSON file.

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - int: Number of vertexes in geojson file
    """
    return len(get_coordinates(file_path)) - 1


def reduce_vertex(file_path: str, ratio: int):
    with fiona.open(file_path) as collection:
       hulls = [concave_hull(shape(feat["geometry"]), ratio) for feat in collection]
        
    dissolved_hulls = mapping(unary_union(hulls))
    
    with open('reduced_vertex.geojson', 'w') as f:
        json.dump(dissolved_hulls, f)


def check_holes(file_path: str) -> bool:
    """
    Checks if GeoJSON has holes by comparing the first and last entry of coordinates.

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - bool: True if there are holes, false if there are no holes
    """
    coordinates_list = get_coordinates(file_path)
    print(coordinates_list)
    return coordinates_list[0] != coordinates_list[-1]

check_holes("overlapping_test.geojson")


def fill_holes(file_path: str):
    """
    Fills holes of GeoJSON by deleting interior ring coordinates and creating a new GeoJSON with new coordinates

    Parameters:
    - file_path: The path to the GeoJSON file.
    """
    coordinates_list = get_coordinates(file_path)

    new_coordinates_list = []
    first_entry = coordinates_list[0]
    new_coordinates_list.append(first_entry)
    for coordinate in coordinates_list[1:]:
        new_coordinates_list.append(coordinate)
        if (coordinate == first_entry):
            break
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        new_coordinates_list
                    ]
                },
                "properties": {}
            }
        ]
    }
    
    with open('filled_holes.geojson', 'w') as f:
        json.dump(geojson, f)


def check_overlap(file_path: str) -> bool:
    """
    Checks if two polygons overlap 

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - bool: True if there is an overlap, false if there is no overlap
    """
    coordinates_list = get_coordinates(file_path)
    polygon1 = []
    polygon2 = []
    first_entry = coordinates_list[0]
    polygon1.append(first_entry)
    divider = 0
    for coordinate in coordinates_list[1:]:
        polygon1.append(coordinate)
        if (coordinate == first_entry):
            divider = coordinates_list[1:].index(coordinate) + 1
            break
    first_entry = coordinates_list[divider]
    for coordinate in coordinates_list[divider+1:]:
        polygon2.append(coordinate)
        if (coordinate == first_entry):
            divider = coordinates_list[1:].index(coordinate) + 1
            break
    
    return Polygon(polygon1).intersects(Polygon(polygon2))
    


    