# PlanetSCA

**PlanetSCA** is an open source python library for mapping snow covered areas
(SCA) from high-resolution PlanetScope images using a Random Forest model. This
package was developed from [original work by Kehan Yang and others](#citations).

This library also include access to a
[pre-trained model](https://huggingface.co/geo-smart/planetsca_models) for
mapping SCA in PlanetScope imagery, and
[sample data](https://huggingface.co/datasets/geo-smart/planetsca_datasets) to
demonstrate the library's functions.

The search and download functions require you to have an account with Planet and
an [API key](https://developers.planet.com/quickstart/apis/#find-your-api-key).

![planetsca flowchart](docs/doc_images/planetsca_flowchart.png)

## Documentation

Please see the **Getting Started** pages of the
[website](https://dshydro.github.io/planetsca/) for installation and basic usage
examples. See the **API Reference** pages for detailed documentation.

## Citations

When using this package, please **cite both the package and the original study
describing the model**:

Citing PlanetSCA:

Citing the original study:

- Yang K., John A., Shean D., Lundquist J.D., Sun Z., Yao F., Todoran S., and
  Cristea N. (2023) High-resolution mapping of snow cover in montane meadows and
  forests using Planet imagery and machine learning. Front. Water 5:1128758.
  doi: [10.3389/frwa.2023.1128758](https://doi.org/10.3389/frwa.2023.1128758)

Other material of interest:

- Code from the original study which was adapted into this library can be found
  [here](https://github.com/KehanGit/High_resolution_snow_cover_mapping).
- A tutorial describing the random forest model in the original study is
  published as a GeoScience Machine Learning Resources and Training (GeoSMART)
  [here](https://geo-smart.github.io/scm_geosmart_use_case/chapters/one.html).
