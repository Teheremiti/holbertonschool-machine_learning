#!/usr/bin/env python3
"""Find schools that have a given topic."""

def schools_by_topic(mongo_collection, topic):
    """Return schools that contain `topic` in their `topics` field.

    Args:
        mongo_collection: A `pymongo.collection.Collection` instance.
        topic: Topic string to search for.

    Returns:
        A list of matching school documents.
    """
    topic_filter = {"topics": {"$in": [topic]}}
    return list(mongo_collection.find(topic_filter))
