import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to URI: {uri}")

# 1. Test raw driver without explicit database target (lets Aura resolve default)
driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    driver.verify_connectivity()
    print("Direct connectivity verified (TLS certificate valid).")
except Exception as e:
    print(f"Connectivity check error: {e}")

# 2. Test querying default session
try:
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        record = result.single()
        print(f"Default session query success: {record['test']}")
except Exception as e:
    print(f"Default session error: {e}")

# 3. Discover available databases (SHOW DATABASES)
try:
    with driver.session() as session:
        result = session.run("SHOW DATABASES")
        dbs = [record["name"] for record in result]
        print(f"Databases found on server: {dbs}")
except Exception as e:
    print(f"SHOW DATABASES query not permitted or failed: {e}")

driver.close()