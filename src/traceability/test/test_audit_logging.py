import pytest
from unittest.mock import MagicMock,patch
from datetime import datetime
import uuid
from exception import CGAgentException
from logger import logging
import src.traceability.audit_logger as audit_logger

# Create deterministic UUID/time
FIXED_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
FIXED_TIME_ISO = "2025-11-15T12:00:00.000000"

#helper sto craete fake db connection
def make_mock_conn(fake_id):
    mock_conn= MagicMock(name="mock_conn")
    mock_cursor=MagicMock(name="mock_cursor")

    #make cursor() usable a scontext manager
    mock_conn.cursor.return_value=mock_cursor

    mock_cursor.fetchone.return_value=[fake_id]

    #allow commit rollback calls
    mock_conn.commit=MagicMock()
    mock_conn.rollback= MagicMock()
    return mock_conn,mock_cursor


@pytest.fixture(autouse=True)
def patch_time_and_uuid():
    with patch("src.traceability.audit_logger.uuid.uuid4", return_value=FIXED_UUID), \
        patch("src.traceability.audit_logger.datetime") as dt_mod:
        
        #also ensure dt_mod.utcnow().isoformat() works if used directly
        dt_mod.utcnow.return_value = datetime.fromisoformat(FIXED_TIME_ISO)
        yield

def test_log_event_calls_gras_and_returns_id(monkeypatch):
    """Happy path: DB insert succeeds and GRAS service is invoked with canonical payload."""
    #arrange : mock DB connection
    mock_conn,mock_cursor=make_mock_conn("returned-event-id")
    
    monkeypatch.setattr(audit_logger,"get_db_connection",lambda:mock_conn)
    
    #patch gars service and enbale flag
    mock_gras= MagicMock(name="mock_gars")
    monkeypatch.setattr(audit_logger,"GRAS_SERVICE",mock_gras)
    monkeypatch.setattr(audit_logger,"GRAS_ENABLED",True)
    
    
    returned_id = audit_logger.log_event(
        agent_name="TestAgent",
        event_type="TEST_WRITE",
        phase="Setup",
        client_transaction_id="TXN123",
        summary="unit test",
        extra_payload={"foo": "bar"},
        trace_id="trace-abc"
    )
    print(f"Event logged successfully with ID: {returned_id}")

    # assert returned id
    assert returned_id == "returned-event-id"

    # fetch GRAS call args **before using them**
    call_args = mock_gras.synthesize_and_ingest_event.call_args[0][0]

    # event_id must NOT be the same as returned-event-id
    assert call_args['event_id'] != "returned-event-id"
    assert len(call_args['event_id']) == 36

    # ensure SQL executed
    mock_conn.commit.assert_called_once()
    mock_cursor.execute.assert_called_once()

    # GRAS assertions
    assert call_args["event_type"] == "TEST_WRITE"
    assert call_args["client_transaction_id"] == "TXN123"
    assert call_args["extra"] == {"foo": "bar"}


def test_log_event_gras_raises_but_sql_persists(monkeypatch):
    """If GRAS raises, SQL commit should still have been attempted and function still returns id."""
    mock_conn, mock_cursor = make_mock_conn("returned-event-id-2")
    monkeypatch.setattr(audit_logger, "get_db_connection", lambda: mock_conn)

    # Make GRAS raise an exception
    mock_gras = MagicMock()
    mock_gras.synthesize_and_ingest_event.side_effect = Exception("graph timeout")
    monkeypatch.setattr(audit_logger, "GRAS_SERVICE", mock_gras)
    monkeypatch.setattr(audit_logger, "GRAS_ENABLED", True)

    eid= audit_logger.log_event(
        agent_name="CriticAgent",
        event_type="POLICY_ENFORCED",
        phase="Phase3",
        client_transaction_id="TXN_FAIL",
        summary="should persist sql despite GRAS error",
        extra_payload=None,
        trace_id="trace-xyz"
    )

    assert eid == 'returned-event-id-2'

    #sql commit should still be called
    mock_conn.commit.assert_called_once()
    #gras was invoked and raised
    assert mock_gras.synthesize_and_ingest_event.called

def test_log_event_db_connection_failure_returns_error(monkeypatch):
    """If get_db_connection raises (no DB), we should return ERROR_NO_DB_CONNECTION."""
    def raise_conn():
        raise CGAgentException("cannot connect",None)
    monkeypatch.setattr(audit_logger,"get_db_connection",raise_conn)
    
    mock_gras = MagicMock(name="mock_gras")
    monkeypatch.setattr(audit_logger, "GRAS_SERVICE", mock_gras)
    monkeypatch.setattr(audit_logger, "GRAS_ENABLED", True)

    #keep gras enabled but it should not be called because db connection fails first
    res = audit_logger.log_event(
        agent_name="DAA",
        event_type="SQL_FAIL_TEST",
        phase="Setup",
        client_transaction_id="TXN_ROLLBACK",
        summary=None,
        extra_payload=None,
        trace_id="trace-no-db"
    )

    assert res == "ERROR_LOGGING_FAILED"

    #GRAS SHOULD NOT BE INVOKED BECAUSE db CONNECTION FAILED EARLY

    assert not mock_gras.synthesize_and_ingest_event.called



def test_log_event_sql_execution_failure_rollbacks(monkeypatch):
    """If cursor.execute raises, log_event should rollback and return ERROR_LOGGING_FAILED."""
    mock_conn,mock_cursor=make_mock_conn("dummy")

    #make exceute rasie
    mock_conn.cursor.return_value.execute.side_effect=Exception("SQL INSERT FAILED")

    monkeypatch.setattr(audit_logger,"get_db_connection",lambda:mock_conn)
    
    
    # Keep GRAS enabled (should not be called because execute raises)
    mock_gras = MagicMock(name="mock_gras")
    monkeypatch.setattr(audit_logger, "GRAS_SERVICE", mock_gras)
    monkeypatch.setattr(audit_logger, "GRAS_ENABLED", True)

    res = audit_logger.log_event(
        agent_name="DAA",
        event_type="SQL_FAIL_TEST",
        phase="Setup",
        client_transaction_id="TXN_ROLLBACK_2",
        summary="sql error path",
        extra_payload=None,
        trace_id="trace-err"
    )

    assert res== "ERROR_LOGGING_FAILED"

    #rollback should be called
    mock_conn.rollback.assert_called_once()

    #gras should not eb called because isnertion failed
    assert not mock_gras.synthesize_and_ingest_event.called
    
    
