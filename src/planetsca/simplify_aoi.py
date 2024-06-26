#https://developers.planet.com/blog/2022/Dec/15/simplifying-your-complex-area-of-interest-a-planet-developers-deep-dive/

import json

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
