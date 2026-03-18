#!/usr/bin/env python3
"""
Find the next SpaceX launch and print its name, date, rocket name, and
launchpad locality.
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


if __name__ == "__main__":
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    launches = get_json(url)
    if launches is None:
        print("Failed to retrieve the next launch.")
        exit(1)

    next_launch = min(launches, key=lambda launch: launch['date_unix'])

    name = next_launch['name']
    date = next_launch['date_local']

    rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
        next_launch['rocket'])
    rocket_info = get_json(rocket_url)
    if rocket_info is None:
        print("Failed to retrieve rocket information.")
        exit(1)

    rocket_name = rocket_info['name']

    launchpad_url = 'https://api.spacexdata.com/v4/launchpads/{}'.format(
        next_launch['launchpad'])
    launchpad_info = get_json(launchpad_url)
    if launchpad_info is None:
        launchpad_name, launchpad_loc = None, None
    else:
        launchpad_name = launchpad_info['name']
        launchpad_loc = launchpad_info['locality']

    if launchpad_name is None or launchpad_loc is None:
        print("Failed to retrieve launchpad information.")
        exit(1)

    print("{} ({}) {} - {} ({})".format(name,
                                        date,
                                        rocket_name,
                                        launchpad_name,
                                        launchpad_loc))
