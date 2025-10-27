---
title: 'PlanetSCA: High-resolution Snow Cover Mapping from Planet Satellite Imagery using Machine Learning'
tags:
  - Python
  - snow cover area
  - machine learning
  - Planet
  - remote sensing
  - hydrology
  - ecology
  - forest gaps
  - mountain meadows
  - water resource management
authors:
  - name: Ian Chiu
    orcid: 0009-0001-0231-5626
    affiliation: 1
  - name: Steven Pestana
    orcid: 0000-0003-3360-0996
    affiliation: 2
  - name: Kehan Yang
    orcid: 0000-0002-1527-7278
    affiliation: "3, 4"
  - name: Emma Boudreau
    orcid: 0009-0000-4395-3512
    affiliation: 1
  - name: Don Setiawan
    orcid: 0000-0002-1624-2667
    affiliation: 1
  - name: Justin Pflug
    orcid: 0000-0002-9604-6307
    affiliation: "3, 5"
  - name: Nicoleta Cristea
    orcid: 0000-0002-9091-0280
    affiliation: 1
affiliations:
  - name: University of Washington, United States
    index: 1
  - name: United States Geological Survey, United States
    index: 2
  - name: Hydrological Sciences Laboratory, NASA Goddard Space Flight Center, United States
    index: 3
  - name: Science Systems and Applications, Inc., United States
    index: 4
  - name: University of Maryland, United States
    index: 5
date: 10 October 2025
bibliography: paper.bib
---

# Summary

PlanetSCA is an open-source Python package that implements a machine learning (random forest) model for mapping snow cover from high-resolution PlanetScope 3–5-m satellite imagery. Traditional snow mapping approaches are based on band indices relying on multispectral satellite data. These indices typically exploit the differences in the reflectance of the visible and shortwave infrared bands (i.e., Normalized Difference Snow Index) to detect snow. However, this method is not applicable to map snow from PlanetScope images, because the images have limited radiometric resolution (mostly red, green, blue, and infrared bands), and only some include shortwave infrared bands. Despite this limitation, we can use a machine learning method and the red, green, blue, and near-infrared bands included in all PlanetScope imagery to map fine-scale snow cover. This random forest model was developed by @Yang:2023 to detect snow-covered areas using limited spectral bands available in PlanetScope images. This method opens up new possibilities for high-resolution (3 m) near-daily snow monitoring in complex terrain, addressing a critical need for accurate snow cover information to support water resource management, ecological research, and climate studies.

PlanetSCA includes functionality such as: data download and search, model training, snow-covered area map generation, and AOI simplification. Sample data and a pre-trained model are provided to demonstrate the library's functions. PlanetSCA requires users to have an account with Planet Data Explore (https://www.planet.com/explorer/) and an API key to utilize its search and download capabilities. Limited access to free PlanetScope images is available at https://www.planet.com/industries/education-and-research/.

# Statement of Need

Information about the timing and extent of snow cover is important for studying mountain meadows, forest ecosystems, and seasonal snow that provides critical water resources [@Sethi:2020; @Breckheimer:2020; @Lowry:2011]. Snowpack in these regions provides essential water storage, yet its distribution is highly heterogeneous, especially in forested areas. Traditional satellite-based methods for snow detection using data from MODIS or VIIRS sensors capture daily snow-covered areas at ~500 m resolution, while moderate resolution imagery (10–30 m), such as from Sentinel-2 and Landsat have less frequent repeat observations, limiting their effectiveness for tracking rapid changes in snow cover at fine spatial resolutions.

Planet satellites, by contrast, capture imagery near daily at 3–5-m resolution, providing fine spatial detail and temporal coverage needed to track small or rapid changes in snow cover. However, the limited spectral information from these satellites, having bands only within the visible and near infrared, has posed challenges for snow detection. PlanetSCA implements the machine learning-based random forest model adapted to Planet's spectral constraints [@Yang:2023]. This approach enables high-resolution, near-daily snow cover mapping across complex terrain where traditional spectral methods cannot resolve fine spatial details of snow cover distribution.

In addition to @Yang:2023, other related research [@Hu:2022; @Cannistra:2021; @John:2022; @Pflug:2024] demonstrated the potential of high-resolution PlanetScope imagery to capture snow cover variations in mountain regions of the Western United States and Switzerland. PlanetSCA not only makes this model accessible for researchers studying hydrology, ecology, or climate but also provides a framework for extending and retraining models for custom applications, and simplifies the workflows through built-in search and download functions for PlanetScope imagery. This functionality positions PlanetSCA as a valuable tool for snow cover mapping and snow and ecology research in regions without extensive ground-based snow observations, where m-scale resolution data are needed.

# Acknowledgements

We acknowledge funding support from NASA grants 80NSSC21K1151 and 80NSSC24K0050, NSF grants EAR−1947875 and OAC−2117834, the University of Washington eScience Institute, and the Scientific Software Engineering Center at the eScience Institute. We would also like to thank Cryocloud for providing us with computing resources and infrastructure for the development of PlanetSCA.

# References
