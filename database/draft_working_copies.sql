-- Current editable contents only. Existing draft/version rows are not changed.
CREATE TABLE IF NOT EXISTS quote_draft_working_copies (
    draft_id UUID PRIMARY KEY REFERENCES quote_drafts(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
