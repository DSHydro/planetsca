import json

import fiona
from shapely.geometry import mapping, shape
from shapely import concave_hull, unary_union
from shapely.geometry import Polygon, mapping



def vertex_count(file_path: str) -> int:
    """
    Counts vertexes from a GeoJSON file.

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - int: Number of vertexes in geojson file
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

    return len(coordinates_list) - 1


def reduce_vertex(file_path: str, ratio: int):
    """
    Reduces the vertex of a given geojson and creates a new geojson with new coordinates

    Parameters:
    - file_path: The path to the GeoJSON file.
    """
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

    first_entry = coordinates_list[0]
    last_entry = coordinates_list[-1]
    
    return first_entry != last_entry


def fill_holes(file_path: str):
    """
    Fills holes of GeoJSON by deleting interior ring coordinates and creating a new GeoJSON with new coordinates

    Parameters:
    - file_path: The path to the GeoJSON file.
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