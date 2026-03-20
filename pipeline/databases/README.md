# Databases

After fetching data (for example from APIs), storing it in a database is a key step in an ML pipeline. This project mixes SQL (relational) and MongoDB (document / NoSQL) examples to create databases, insert data, and query it for processing.

## What's included

### SQL (relational)
The `.sql` scripts numbered `0` to `21` are SQL queries for the relational part of the project (see `0-create_database_if_missing.sql`, which creates `db_0`).

### MongoDB (document / NoSQL)
Scripts `22` to `29` are MongoDB shell scripts, and scripts `30` to `34` are Python scripts using `pymongo`.

## Quick run notes

- SQL: run the `.sql` files with your SQL client (MySQL in this project).

- MongoDB shell: run scripts `22` to `29` with `mongo`.

- Python/PyMongo: run scripts `30` to `34` with `python3` (expects MongoDB at `mongodb://localhost:27017/`).