#!/usr/bin/env python3
"""
Print basic statistics for Nginx access logs stored in MongoDB.
"""

from pymongo import MongoClient


def get_nginx_stats():
    """Retrieve and print Nginx log statistics from MongoDB.

    Connects to the local `logs.nginx` collection.

    Returns:
        None.
    """
    mongo_client = MongoClient("mongodb://localhost:27017/")
    logs_database = mongo_client.logs
    nginx_logs_collection = logs_database.nginx

    # Total logs
    total_logs = nginx_logs_collection.count_documents({})
    print(f"{total_logs} logs")

    # Per-method counts
    print("Methods:")
    http_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for http_method in http_methods:
        method_count = nginx_logs_collection.count_documents({"method": http_method})
        print(f"\tmethod {http_method}: {method_count}")

    # Health-check requests
    status_check = nginx_logs_collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print(f"{status_check} status check")


if __name__ == "__main__":
    get_nginx_stats()
