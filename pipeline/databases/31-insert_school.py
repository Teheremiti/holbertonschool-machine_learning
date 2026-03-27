#!/usr/bin/env python3
"""Insert a document into a MongoDB collection."""

def insert_school(mongo_collection, **kwargs):
    """Insert a new document into `mongo_collection`.

    Args:
        mongo_collection: A `pymongo.collection.Collection` instance.
        kwargs: Fields to insert into the document.

    Returns:
        The inserted document id (`inserted_id`).
    """
    insert_result = mongo_collection.insert_one(kwargs)
    return insert_result.inserted_id
