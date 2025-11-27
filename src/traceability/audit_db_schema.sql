-- db/audit_db_schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- CORE LOGGING TABLE: audit_event
-- Purpose: Records every action/decision made by the agents.
-- Every insertion is an event; never UPDATE content fields (treat as append-only)
-- db/audit_db_schema_FIXED.sql
-- Recreate with proper structure:

-- 1. CORE LOGGING TABLE: audit_event (UNCHANGED but add indexes)
CREATE TABLE audit_event (
  event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  trace_id CHAR(32) NULL,
  client_transaction_id TEXT NULL,
  agent_name TEXT NOT NULL,
  phase TEXT NOT NULL,
  event_type TEXT NOT NULL,
  summary TEXT NULL,
  extra_payload JSONB NULL
);

-- ADD INDEXES for performance
CREATE INDEX idx_audit_event_trace_id ON audit_event(trace_id);
CREATE INDEX idx_audit_event_client_txn ON audit_event(client_transaction_id);
CREATE INDEX idx_audit_event_agent_phase ON audit_event(agent_name, phase);

-- 2. GOVERNANCE TABLE: policy_document (UNCHANGED)
CREATE TABLE policy_document (
  policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  policy_name TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  content_sha256 CHAR(64) NOT NULL,
  storage_path TEXT NOT NULL,
  s3_path  TEXT NULL,
); 

-- 3. DATA INTEGRITY TABLE: dataset_version (FIXED - ADD UUID PRIMARY KEY)
CREATE TABLE dataset_version (
  dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),  -- ✅ ADDED PRIMARY KEY
  dataset_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  split_group TEXT NOT NULL,
  sha256 CHAR(64) NOT NULL,
  dvc_rev TEXT NULL,
  dvc_path TEXT NULL,
  s3_path  TEXT NULL,
  row_count INTEGER NULL,
  col_count INTEGER NULL,
  
  -- ADD unique constraint to prevent duplicate versions
  UNIQUE(dataset_version, sha256)
);

-- 4. EVIDENCE POINTERS: evidence_pointer (UNCHANGED)
CREATE TABLE evidence_pointer (
  evidence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  kind TEXT NOT NULL,
  dvc_rev TEXT NULL,
  dvc_path TEXT NOT NULL,
  s3_path  TEXT NULL,
  sha256 CHAR(64) NULL
);

-- 5. LINKING TABLE: event_link (NOW VALID - references fixed dataset_version)
CREATE TABLE event_link (
  link_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  event_id UUID NOT NULL REFERENCES audit_event(event_id),
  policy_id UUID NULL REFERENCES policy_document(policy_id),
  dataset_id UUID NULL REFERENCES dataset_version(dataset_id), 
  evidence_id UUID NULL REFERENCES evidence_pointer(evidence_id),
  
  -- ADD constraint to ensure at least one link exists
  CHECK (
    policy_id IS NOT NULL OR 
    dataset_id IS NOT NULL OR 
    evidence_id IS NOT NULL
  )
);

-- ADD INDEXES for event_link foreign keys
CREATE INDEX idx_event_link_event ON event_link(event_id);
CREATE INDEX idx_event_link_dataset ON event_link(dataset_id);
CREATE INDEX idx_event_link_policy ON event_link(policy_id);
CREATE INDEX idx_event_link_evidence ON event_link(evidence_id);
- 2. For TRAINING pipeline queries
CREATE INDEX idx_audit_event_trace_id ON audit_event(trace_id);
CREATE INDEX idx_audit_event_agent_phase ON audit_event(agent_name, phase);

-- 3. For PREDICTION pipeline queries (when clients exist)
CREATE INDEX idx_audit_event_client_txn ON audit_event(client_transaction_id);