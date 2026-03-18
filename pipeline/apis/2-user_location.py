#!/usr/bin/env python3
"""
Query a GitHub API URL and print the user's `location`.

For HTTP 403 responses, it also prints the reset time computed from the
`X-Ratelimit-Reset` header.
"""
import requests
import sys
import time


def handle_http_error(e):
    """Handle HTTP errors."""
    if e.response.status_code == 403:
        rate_lim = int(e.response.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_lim - now) / 60)
        print("Reset in {} min".format(minutes))
    elif e.response.status_code == 404:
        print("Not found")
    else:
        print("HTTP error: {}".format(e))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit()

    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github+json'}

    try:
        results = requests.get(url, headers=headers)
        results.raise_for_status()
    except requests.exceptions.HTTPError as e:
        handle_http_error(e)
        exit(1)
    except requests.exceptions.RequestException as e:
        print("Request failed: {}".format(e))
        exit(1)

    print(results.json()["location"])
