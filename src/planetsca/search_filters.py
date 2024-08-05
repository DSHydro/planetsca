import json

from shapely.geometry import shape


def make_domain_geometry_from_bounds(bounds: list[float]):
    """
    Helper - Make a shapely geometry polygon from from longitude and latitude bounds (a rectangular area)

    Parameters
    ----------
        bounds: list[float]
            longitude and latitude coordinates of the area, in the order [minLon, minLat, maxLon, maxLat]
    Returns
    -------
        geo_json_geometry: dict
            a geojson-like dictionary
        domain_geometry: shapely.geometry.polygon.Polygon
            shaply geometry polygon defined by these bounds
    """

    [minLon, minLat, maxLon, maxLat] = bounds

    # create a geojson-like geometry dictionary
    geo_json_geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [minLon, minLat],  # lower left corner
                [maxLon, minLat],  # lower right corner
                [maxLon, maxLat],  # upper right corner
                [minLon, maxLat],  # upper left corner
                [minLon, minLat],  # lower left corner again
            ]
        ],
    }

    # create a shapely geometry
    domain_geometry = shape(geo_json_geometry)

    return geo_json_geometry, domain_geometry


def make_geometry_filter_from_bounds(bounds: list[float]) -> dict:
    """
    Make a geometry filter dictionary for the Planet API from longitude and latitude bounds (a rectangular search area)

    Parameters
    ----------
        bounds: list[float]
            longitude and latitude coordinates of the area, in the order [minLon, minLat, maxLon, maxLat]
    Returns
    -------
        geometry_filter: dict
            dictionary geometry filter for the Planet API
        geometry: shapely.geometry.polygon.Polygon
            shaply geometry polygon defined by these bounds
    """

    # create a geojson-like geometry dictionary
    geo_json_geometry, _ = make_domain_geometry_from_bounds(bounds)

    # create the geometry filter for the Planet API
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geo_json_geometry,
    }

    return geometry_filter


def make_geometry_filter_from_geojson(geo_json_path: str) -> dict:
    """
    Make a geometry filter dictionary for the Planet API from a geojson file path

    Parameters
    ----------
        geo_json_path: str
            Path to the geojson file to be turned into a filter
    Returns
    -------
        geometry_filter: dict
            dictionary geometry filter for the Planet API
    """

    with open("geo_json_path", "r") as file:
        geojson_data = json.load(file)

    coords = []

    coords.extend(
        [coord for polygon in geojson_data["coordinates"] for coord in polygon]
    )

    geo_json_geometry = {
        "type": "Polygon",
        "coordinates": [coords],
    }

    # create the geometry filter for the Planet API
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geo_json_geometry,
    }

    return geometry_filter


def make_date_range_filter(start_date: str, end_date: str) -> dict:
    """
    Make a date range filter dictionary for the Planet API

    Parameters
    ----------
        start_date: str
            start date (UTC) in the format 'YYYY-mm-ddTHH:MM:SSZ' (e.g. '2023-07-25T00:00:00Z')
        end_date: str
            end date (UTC) in the format 'YYYY-mm-ddTHH:MM:SSZ' (e.g. '2023-07-25T00:00:00Z')
    Returns
    -------
        date_range_filter: dict
            dictionary date range filter for the Planet API
    """

    # filter images acquired in a certain date range
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {"gte": start_date, "lte": end_date},
    }

    return date_range_filter


def make_cloud_cover_filter(lte: float, gte: float = 0) -> dict:
    """
    Make a cloud cover filter dictionary for the Planet API

    Parameters
    ----------
        lte: float
            cloud cover less than or equal to
        gte: float
            cloud cover greater than or equal to value, defaults to 0
    Returns
    -------
        cloud_cover_filter: dict
            dictionary cloud cover filter for the Planet API
    """

    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {"gte": gte, "lte": lte},
    }

    return cloud_cover_filter


def make_AndFilter(filters: list[dict]) -> dict:
    """
    Make a filter by combining two or more filters with an AND operator

    Parameters
    ----------
        filters: list[dict]
            list of filters to combine with AND
    Returns
    -------
        filter: dict
            combined filters for the Planet API
    """

    filter = {
        "type": "AndFilter",
        "config": filters,
    }
    return filter


def make_OrFilter(filters: list[dict]) -> dict:
    """
    Make a filter by combining two or more filters with an OR operator

    Parameters
    ----------
        filters: list[dict]
            list of filters to combine with OR
    Returns
    -------
        filter: dict
            combined filters for the Planet API
    """

    filter = {
        "type": "OrFilter",
        "config": filters,
    }
    return filter
