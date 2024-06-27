import json
import fiona
from shapely.geometry import mapping, shape
from shapely import concave_hull, unary_union
from shapely.geometry import Polygon, mapping
import os


def get_coordinates(file_path: str) -> list:
    """
    Gets coordinates from a GeoJSON file - Helper Method

    Parameters:
    - file_path: The path to the GeoJSON file

    Returns:
    - list: List of vertices from GeoJSON polygon
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
    """
    Reduces the number of vertices in a polygon

    Parameters:
    - file_path: The path to the GeoJSON file
    - ratio: Sets the concave_hull ratio
    """
    with fiona.open(file_path) as collection:
       hulls = [concave_hull(shape(feat["geometry"]), ratio) for feat in collection]
        
    dissolved_hulls = mapping(unary_union(hulls))
    
    with open('reduced_vertex.geojson', 'w') as f:
        json.dump(dissolved_hulls, f)


def check_hole(file_path: str) -> bool:
    """
    Checks if a polygon contains a hole.

    Parameters:
    - file_path: The path to the GeoJSON file.

    Returns:
    - bool: True if there is a hole in the polygon, false if there is not
    """
    with open(file_path) as f:
        geojson = json.load(f)

    for feature in geojson['features']:
        if feature['geometry']['type'] == 'Polygon':
            coordinates = feature['geometry']['coordinates']
            if len(coordinates) > 1:
                return True
    return False


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
    Checks if two polygons are overlapping from a GeoJSON file

    Parameters:
    - file_path: The path to the GeoJSON file.

     Returns:
    - bool: True if there is overlap and false if there is no overlap
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


def fix_overlap(file_path: str):
    """
    Fixes the overlap of two polygons by deleting the overlapped area and merging the two polygons into one polygon

    Parameters:
    - file_path: The path to the GeoJSON file.
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
    combined_polygon = unary_union([Polygon(polygon1), Polygon(polygon2)])

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(combined_polygon),
                "properties": {}
            }
        ]
    }
    
    with open('corrected_overlap.geojson', 'w') as f:
        json.dump(geojson, f)


def clipping_check(file_path: str, AOI_Coordinates: list) -> bool:
    """
    Checks if a given polygon clips outside the contracted AOI coordinates

    Parameters:
    - file_path: The path to the GeoJSON file
    - AOI_Coordinates: List of coordinates representing the AOI bounds

     Returns:
    - bool: True if it clips outside and false if it does not
    """
    given_polygon = Polygon(get_coordinates(file_path))

    first_array = AOI_Coordinates[0]
    last_array = AOI_Coordinates[-1]

    if first_array != last_array:
        AOI_Coordinates.append(first_array.copy())


    AOI_Bounds = Polygon(AOI_Coordinates)

    return not given_polygon.within(AOI_Bounds)


def fix_clipping(file_path: str, AOI_Coordinates: list):
    """
    Deletes all parts of a Polygon that exceed the AOI coordinate bounds

    Parameters:
    - file_path: The path to the GeoJSON file
    - AOI_Coordinates: List of coordinates representing the AOI bounds
    """

    first_array = AOI_Coordinates[0]
    last_array = AOI_Coordinates[-1]

    if first_array != last_array:
        AOI_Coordinates.append(first_array.copy())
    
    combined_polygon = Polygon(AOI_Coordinates).intersection(Polygon(get_coordinates(file_path)))

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(combined_polygon),
                "properties": {}
            }
        ]
    }
    
    with open('corrected_clipping.geojson', 'w') as f:
        json.dump(geojson, f)


def simplify(file_path: str, ratio: int, AOI_Coordinates: list):
    """
    Runs all checks and fixes to adhere to Planet's AOI Geometry Limits

    Parameters:
    - file_path: The path to the GeoJSON file
    - AOI_Coordinates: List of coordinates representing the AOI bounds
    """
    update_file_path = file_path
    #Vertices Check
    if (vertex_count(file_path) > 500):
        print("Too many vertices, reducing number")
        reduce_vertex(update_file_path)
        update_file_path = 'reduced_vertex.geojson'
        
    #Holes and Exterior Rings Check
    if (check_hole(update_file_path)):
        print("Polygon contains hole, filling hole")
        fill_holes(update_file_path)
        update_file_path = 'filled_holes.geojson'
        
    #Overlapping and Intersections Check
    if (check_overlap(update_file_path)):
        print("Polygons are overlapping, combining polygons to eliminate overlap")
        fix_overlap(update_file_path)
        update_file_path = 'corrected_overlap.geojson'
        
    #Clipping outside AOI Check
    if (clipping_check(update_file_path, AOI_Coordinates)):
        print("Polygon clips outside of AOI, cutting outside areas")
        fix_clipping(update_file_path, AOI_Coordinates)
        update_file_path = 'corrected_clipping.geojson'

    try:
        os.rename(update_file_path, "Simplified.geojson")
        print("Your new simplified file is called Simplified.geojson. For more information on Planet's AOI Geometry Limits, check out this link: https://developers.planet.com/docs/subscriptions/tools/")
    except PermissionError:
        print("Permission denied.")
    
    if os.path.exists('reduced_vertex.geojson'):
        os.remove('reduced_vertex.geojson')
    if os.path.exists('filled_holes.geojson'):
        os.remove('filled_holes.geojson')
    if os.path.exists('corrected_overlap.geojson'):
        os.remove('corrected_overlap.geojson')
        