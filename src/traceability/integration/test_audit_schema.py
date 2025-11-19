import os
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor


SCHEMA_PATH=r"src\traceability\audit_db_schema.sql" 

#FCITIRE :POSTGRESQL CONNECTION
@pytest.fixture(scope="session")
def db_conn():
    """
    Creates a real PostgreSQL connection for schema validation.
    Assume

    """
    conn=psycopg2.connect(dbname="audit_db",user="test",password="test",host="localhost",port=5432)
    conn.autocommit=True
    yield conn
    conn.close()


@pytest.fixture(scope="session",autouse=True)
def apply_schema(db_conn):
    """ apply audit_db_schema.sql before running tests"""
    with db_conn.cursor() as cur:
        with open(SCHEMA_PATH,"r") as f:
            sql= f.read()
            cur.execute(sql)

# test 1 verify tables created

def test_tables_exist(db_conn):
    expected_tables=[
        "audit_event",
        "policy_document",
        "dataset_version",
        "evidence_pointer",
        "event_link"
    ]

    with db_conn.cursor() as cur:
        cur.execute(""" SELECT table_name FROM information_schema.tables
        WHERE table_schema='public';""")
        tables= [r[0] for r in cur.fetchall()]

    for table in expected_tables:
        assert table in tables,f"Table missing:{table}"


# test 2 verify columns for audit_event

def test_audit_event_columns(db_conn):
    expected_cols={
        "event_id",
        "created_at",
        "trace_id",
        "client_transaction_id",
        "agent_name",
        "phase",
        "event_type",
        "summary",
        "extra"
    }

    with db_conn.cursor() as cur:
        cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name='audit_event';""")

        cols=[r[0] for r in cur.fetchall()]

    missing= expected_cols-set(cols)
    assert not missing,f"Missing columns:{missing}"
#tets 3 tets simple inserts

def test_insert_into_all_tables(db_conn):
    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:

        #insert into policy document
        cur.execute("""
        INSERT INTO policy_document(policy_name,policy_version,content_sha256,storage_path)
        VALUES('DATA-TEST','1.0',repeat('A',64),'path/test.yaml')
        RETURNING policy_id
        """)
       
        policy_id= cur.fetchone()['policy_id']

        #insert into dataset_version
        cur.execute("""
        INSERT INTO dataset_version(split_group, sha256, dvc_rev, dvc_path, row_count, col_count)
            VALUES ('train+val', repeat('B',64), 'rev123', 'dvc/path', 100, 20)
            RETURNING dataset_id;
        """)

        dataset_id = cur.fetchone()["dataset_id"]

        # Insert into evidence_pointer
        cur.execute("""
            INSERT INTO evidence_pointer(kind, dvc_path, sha256)
            VALUES ('MODEL_CARD', 'evidence/model.pdf', repeat('C',64))
            RETURNING evidence_id;
        """)
        evidence_id = cur.fetchone()["evidence_id"]

        # Insert into audit_event
        cur.execute("""
            INSERT INTO audit_event(agent_name, phase, event_type, summary, extra)
            VALUES ('DAA', 'Phase1', 'POLICY_CHECK', 'test summary', '{"key": "value"}')
            RETURNING event_id;
        """)
        event_id = cur.fetchone()["event_id"]

        # Insert into event_link with FK references
        cur.execute("""
            INSERT INTO event_link(event_id, policy_id, dataset_id, evidence_id)
            VALUES (%s, %s, %s, %s)
            RETURNING link_id;
        """, (event_id, policy_id, dataset_id, evidence_id))

        link_id = cur.fetchone()["link_id"]

        assert event_id is not None
        assert policy_id is not None
        assert dataset_id is not None
        assert evidence_id is not None
        assert link_id is not None


#test 4 foreign key enforcement

def test_fk_enforcement(db_conn):
    with db_conn.cursor() as cur:
        with pytest.raises(Exception):
            #insert invalid FK(policy_id does not exist)
            cur.execute("""
            INSERT INTO event_link(event_id,policy_id)
            VALUES(UUID_GENERATE_V4(),uuid_generate_v4());
            """)

