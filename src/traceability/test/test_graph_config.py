import pytest
from src.traceability.graph_config import GraphDBClient, MOCK_DB_STATE
from exception import CGAgentException


@pytest.fixture
def client():
    return GraphDBClient()


# WRITE OPERATIONS

def test_write_merge(client):
    query = "MERGE (n:Test {id:'TXN-101'})"
    res = client.execute_cypher(query)
    assert res == {"status": "success", "operation": "write"}


def test_write_create(client):
    query = "CREATE (n:Test {id:'TXN-999'})"
    res = client.execute_cypher(query)
    assert res == {"status": "success", "operation": "write"}



# READ: FinalDecision

def test_read_finaldecision(client):
    query = "MATCH (n:FinalDecision {txn:'TXN-101'}) RETURN n"
    res = client.execute_cypher(query)

    expected = {
        'd': MOCK_DB_STATE["TXN-101"]['d'],
        'violations': MOCK_DB_STATE["TXN-101"]['violations'],
        'policies': MOCK_DB_STATE["TXN-101"]['policies']
    }

    assert res == expected


def test_read_finaldecision_unknown_txn(client):
    query = "MATCH (n:FinalDecision {txn:'TXN-999'}) RETURN n"
    res = client.execute_cypher(query)
    assert res == {"status": "unknown"}


# READ: DatasetVersion

def test_read_datasetversion(client):
    query = "MATCH (d:DatasetVersion {txn:'TXN-101'}) RETURN d"
    res = client.execute_cypher(query)
    assert res == {'ds.hash': MOCK_DB_STATE["TXN-101"]['ds.hash']}


# READ: Artifact
def test_read_artifact(client):
    query = "MATCH (a:Artifact {txn:'TXN-101'}) RETURN a"
    res = client.execute_cypher(query)

    assert res == {
        'art.id': MOCK_DB_STATE["TXN-101"]['art.id'],
        'art.kind': MOCK_DB_STATE["TXN-101"]['art.kind']
    }


# LOWERCASE QUERY SUPPORT

def test_lowercase_query(client):
    query = "match (a:Artifact {txn:'TXN-101'}) return a"
    res = client.execute_cypher(query)

    assert res == {
        'art.id': MOCK_DB_STATE["TXN-101"]['art.id'],
        'art.kind': MOCK_DB_STATE["TXN-101"]['art.kind']
    }


# BAD REGEX → SHOULD RAISE ERROR

def test_invalid_regex_raises(client):
    # break regex intentionally
    query = "MATCH (n:FinalDecision {txn:TXN-101}) RETURN n"  # missing quotes

    with pytest.raises(CGAgentException):
        client.execute_cypher(query)


# UNKNOWN QUERY TYPE

def test_unknown_query(client):
    query = "MATCH (n:RandomNode {txn:'TXN-101'}) RETURN n"
    res = client.execute_cypher(query)
    assert res == {"status": "unknown"}
