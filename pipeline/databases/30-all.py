#!/usr/bin/env python3
"""List all documents from a MongoDB collection."""

def list_all(mongo_collection):
    """Return all documents in `mongo_collection`.

    Args:
        mongo_collection: A `pymongo.collection.Collection` instance.

    Returns:
        A list of documents. Empty if the collection has no documents.
    """
    documents_cursor = mongo_collection.find({})
    return list(documents_cursor)
