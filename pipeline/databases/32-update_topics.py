#!/usr/bin/env python3
"""Update the topics of a school document."""

def update_topics(mongo_collection, name, topics):
    """Update all topics for schools matching `name`.

    Args:
        mongo_collection: A `pymongo.collection.Collection` instance.
        name: School name to update.
        topics: List of topics to store.

    Returns:
        None.
    """
    filter_query = {"name": name}
    update_query = {"$set": {"topics": topics}}
    mongo_collection.update_many(filter_query, update_query)
