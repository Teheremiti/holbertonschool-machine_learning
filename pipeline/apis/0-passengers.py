#!/usr/bin/env python3
"""
Find SWAPI starships that can carry at least a given number of passengers.

This script queries the paginated `/api/starships/` endpoint and returns the
names of ships that meet the passenger count threshold.
"""
import requests


def availableShips(passengerCount):
    """
    Return starships that can accommodate at least passengerCount people.

    Args:
        passengerCount (int): Minimum number of passengers.

    Returns:
        List[str]: Names of ships with capacity >= passengerCount.
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        try:
            r = requests.get(url).json()
        except requests.exceptions.RequestException as e:
            print("Request failed: {}".format(e))
            return []
        ships += [ship for ship in r.get("results") if ship.get("passengers") != "n/a"
                  and ship.get("passengers") != "unknown"
                  and int(ship.get("passengers").replace(",", "")) >= passengerCount]
        url = r.get("next")

    return [ship.get("name") for ship in ships]
