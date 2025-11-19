-- db/audit_db_schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ----------------------------------------------------------------------
-- CORE LOGGING TABLE: audit_event
-- Purpose: Records every action/decision made by the agents.
-- Every insertion is an event; never UPDATE content fields (treat as append-only)
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_event (
  event_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

  -- CRITICAL ADDITION: Links the log event to a specific client or transaction.
  trace_id               CHAR(32)         NULL, -- OTel Trace ID (32 hex chars)
  client_transaction_id  TEXT             NULL, -- e.g., Client ID, Request ID, or Trace ID
  
  agent_name             TEXT             NOT NULL,  -- e.g., 'DAA', 'CriticAgent'
  phase                  TEXT             NOT NULL,  -- e.g., 'Phase1_DataAssurance', 'Phase3_RuntimeAudit'
  event_type             TEXT             NOT NULL,  -- e.g., 'POLICY_CHECK', 'DECISION_FINALIZED', 'DATA_QUALITY_FLAG'
  
  -- The core decision/summary of the event
  summary                TEXT             NULL,      -- human-readable short description (e.g., "Policy check failed: Prohibited Attribute")
  extra                  JSONB            NULL       -- Arbitrary context (e.g., XAI feature scores, specific drift metrics)
);

-- ----------------------------------------------------------------------
-- GOVERNANCE TABLE: policy_document
-- Purpose: Stores the immutable declaration of governance rules (PaC Metadata).
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS policy_document (
  policy_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
  policy_name            TEXT             NOT NULL,  -- e.g., 'DATA-ASSURANCE-1.0', 'RUNTIME-AUDIT-2.1'
  policy_version         TEXT             NOT NULL,  -- '1.0'
  
  -- MODIFICATION: Content stored as a hash. Auditor uses this hash to retrieve the
  -- actual YAML file from the versioned archive (DVC).
  content_sha256         CHAR(64)         NOT NULL,  -- Hash of the YAML bytes
  
  storage_path           TEXT             NOT NULL   -- DVC path (e.g., data/version/metadata.yaml)
);

-- ----------------------------------------------------------------------
-- DATA INTEGRITY TABLE: dataset_version
-- Purpose: Logs the immutable integrity hash of all certified training data partitions.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_version (
  dataset_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
  split_group            TEXT             NOT NULL,  -- 'train+val' or 'test'
  sha256                 CHAR(64)         NOT NULL,  -- The forensic hash key
  
  -- DVC retrieval pointers
  dvc_rev                TEXT             NULL,      -- Git commit associated with the DVC version
  dvc_path               TEXT             NULL,      -- path inside repo
  
  row_count              INTEGER          NULL,
  col_count              INTEGER          NULL
);

-- ----------------------------------------------------------------------
-- EVIDENCE POINTERS: evidence_pointer
-- Purpose: Provides retrieval links to large, archived evidence files.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_pointer (
  evidence_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
  kind                   TEXT             NOT NULL,  -- e.g., 'CONFIDENT_LEARNING_REPORT', 'MODEL_CARD_PDF'
  dvc_rev                TEXT             NULL,
  dvc_path               TEXT             NOT NULL,  -- The DVC path where the auditor can fetch the file
  sha256                 CHAR(64)         NULL       -- Hash of the evidence file contents
);

-- ----------------------------------------------------------------------
-- LINKING TABLE: event_link (Soft foreign keys)
-- Purpose: Links events in the audit log to the governance artifacts they referenced.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_link (
  link_id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
  event_id               UUID             NOT NULL REFERENCES audit_event(event_id),
  policy_id              UUID             NULL REFERENCES policy_document(policy_id),
  dataset_id             UUID             NULL REFERENCES dataset_version(dataset_id),
  evidence_id            UUID             NULL REFERENCES evidence_pointer(evidence_id)
);