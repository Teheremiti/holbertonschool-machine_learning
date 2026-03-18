#!/usr/bin/env python3
"""
Count launches per rocket and print them in descending order.

This script fetches SpaceX launches, looks up each rocket's name, and
counts how often each rocket appears.
"""
import requests


def get_json(url):
    """Fetch JSON data from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print("Request failed: {}".format(e))
        return None


def countLaunchesPerRocket(launches):
    """Count launches per rocket name."""
    rocket_dict = {}

    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
            rocket_id)

        rocket_info = get_json(rocket_url)
        rocket_name = rocket_info.get('name')

        if rocket_dict.get(rocket_name) is None:
            rocket_dict[rocket_name] = 1
            continue
        rocket_dict[rocket_name] += 1

    return rocket_dict


if __name__ == "__main__":
    url = 'https://api.spacexdata.com/v4/launches'
    launches = get_json(url)
    rocket_dict = countLaunchesPerRocket(launches)
    sorted_rocket = sorted(rocket_dict.items(),
                           key=lambda kv: kv[1],
                           reverse=True)
    for rocket, count in sorted_rocket:
        print("{}: {}".format(rocket, count))
