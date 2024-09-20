from typing import List, Literal

import geopandas as gpd
import requests
from requests.auth import HTTPBasicAuth
from shapely.geometry import shape

from planetsca import simplify_aoi


def search(api_key: str, filter: dict, item_type: str = "PSScene") -> gpd.GeoDataFrame:
    """
    Sends a request to the Planet API to find if data is available

    Parameters
    ----------
        api_key: str
            Planet API key
        filter: dict
            Dictionary containing data filter information
        item_type: str
            Class of spacecraft and/or processing level of an item, defaults to PSScene. See https://developers.planet.com/docs/apis/data/items-assets/

    Returns
    -------
        gdf: geopandas.geodataframe.GeoDataFrame
            GeoDataFrame containing information about the Planet images returned by the search
    """

    # Search API request object
    search_endpoint_request = {"item_types": [item_type], "filter": filter}
    response = requests.post(
        "https://api.planet.com/data/v1/quick-search",
        auth=HTTPBasicAuth(api_key, ""),
        json=search_endpoint_request,
    )

    # check to make sure we received items in the response
    if len(response.json()["features"]) > 0:
        print(f"Search returned {len(response.json()['features'])} items.")
        # use the response from the Planet API to make a geodataframe of the returned image IDs and info
        gdf = response_to_gdf(response, filter)
        return gdf
    else:
        print(
            f"Search returned {len(response.json()['features'])} items. Try changing search filters"
        )
        return None


def response_to_gdf(response: requests.Response, filter: dict):
    """
    Creates geodataframe of image IDs and other information from a Planet API response

    Parameters
    ----------
        response: requests.Response
            Response object from the Planet API containing information about images that matched search criteria
        filter: dict
            Dictionary containing data filter information

    Returns
    -------
        gdf: geopandas.geodataframe.GeoDataFrame
            GeoDataFrame containing information about the Planet images returned by the search
    """

    domain_geometry = shape(get_filter(filter, "GeometryFilter")["config"])

    # view available data and prepare the list of planet IDs to download
    geojson_data = response.json()
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    # Add a new column to 'gdf' with the intersection area
    gdf["intersection_area"] = gdf["geometry"].intersection(domain_geometry).area

    # Calculate the percentage overlap
    gdf["overlap_percentage"] = (gdf["intersection_area"] / domain_geometry.area) * 100

    # get image IDs and add to geodataframe
    gdf["id"] = [feature["id"] for idx, feature in enumerate(geojson_data["features"])]

    return gdf


def make_domain_geometry_from_bounds(bounds: List[float]):
    """
    Make a shapely geometry polygon from from longitude and latitude bounds (a rectangular area)

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


def make_geometry_filter_from_bounds(bounds: List[float]) -> dict:
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


def make_geometry_filter_from_geojson(geojson_filepath: str) -> dict:
    """
    Make a geometry filter dictionary for the Planet API from a geojson file path

    Parameters
    ----------
        geojson_filepath: str
            Path to the geojson file to be turned into a filter
    Returns
    -------
        geometry_filter: dict
            dictionary geometry filter for the Planet API
    """

    coords = simplify_aoi.get_coordinates(geojson_filepath)

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


def combine_filters(
    filters: List[dict], combine_with: Literal["and", "or"] = "and"
) -> dict:
    """
    Combine two or more filters with either an AND, or an OR operator

    Parameters
    ----------
        filters: List[dict]
            list of filters to combine with AND
        combine_with:  Literal['and', 'or']
            specify whether to combine filters with an AND, or an OR operator (defaults to AND)
    Returns
    -------
        filter: dict
            combined filters for the Planet API
    """
    if combine_with == "and":
        type = "AndFilter"
    elif combine_with == "or":
        type = "OrFilter"
    filter = {
        "type": type,
        "config": filters,
    }
    return filter


def get_filter(
    filter: dict,
    filter_type: Literal[
        "GeometryFilter", "DateRangeFilter", "RangeFilter", "AndFilter", "OrFilter"
    ],
):
    """
    Return an individual filter from a dictionary containing multiple Planet API search filters created with search_filters.combine_filters()

    Parameters
    ----------
    filter: dict
        Filter dict
    filter_type:  Literal["GeometryFilter", "DateRangeFilter", "RangeFilter", "AndFilter", "OrFilter"]
        Specify which sub-filter we want to return
    Returns
    -------
    sub_filter: dict
        sub-filter within the larger filter dict
    """

    sub_filter = next(item for item in filter["config"] if item["type"] == filter_type)

    return sub_filter
