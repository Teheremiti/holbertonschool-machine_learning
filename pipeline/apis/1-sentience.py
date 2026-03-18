#!/usr/bin/env python3
"""
List homeworld names for sentient species from the SWAPI API.

This script fetches all species, filters those classified or designated as
'sentient', then returns each species' homeworld name.
"""
import requests


def sentientSpecies():
    """
    Return sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    species = []
    while url:
        try:
            r = requests.get(url).json()
        except requests.exceptions.RequestException as e:
            print("Request failed: {}".format(e))
            return []
        species += [
            species
            for species in r.get("results", [])
            if (species.get("designation") == "sentient"
                or species.get("classification") == "sentient")
            and species.get("homeworld") is not None
        ]
        url = r.get("next")
    return species


def sentientPlanets():
    """
    Return homeworld names for sentient species.

    Returns:
        List[str]: Planet names for species whose classification or
        designation is 'sentient'.
    """
    sentient_species = sentientSpecies()
    planets = []
    for species in sentient_species:
        url = species.get("homeworld")
        if url is not None:
            try:
                r = requests.get(url).json()
            except requests.exceptions.RequestException as e:
                print("Request failed: {}".format(e))
                planets.append("unknown")
                continue
            planets.append(r.get("name"))
        else:
            planets.append("unknown")
    return planets
