CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION contact_management_trim_text()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_TABLE_NAME = 'client_contacts' THEN
        NEW.hubspot_record_id = NULLIF(BTRIM(COALESCE(NEW.hubspot_record_id, '')), '');
        NEW.company_name = BTRIM(COALESCE(NEW.company_name, ''));
        NEW.first_name = BTRIM(COALESCE(NEW.first_name, ''));
        NEW.last_name = BTRIM(COALESCE(NEW.last_name, ''));
    ELSIF TG_TABLE_NAME = 'internal_contacts' THEN
        NEW.name = BTRIM(COALESCE(NEW.name, ''));
        NEW.title = BTRIM(COALESCE(NEW.title, ''));
    END IF;

    NEW.email = LOWER(BTRIM(COALESCE(NEW.email, '')));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION contact_management_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS client_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hubspot_record_id TEXT NULL,
    company_name TEXT NOT NULL CHECK (BTRIM(company_name) <> ''),
    first_name TEXT NOT NULL CHECK (BTRIM(first_name) <> ''),
    last_name TEXT NOT NULL CHECK (BTRIM(last_name) <> ''),
    email TEXT NOT NULL CHECK (BTRIM(email) <> ''),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS internal_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL CHECK (BTRIM(name) <> ''),
    title TEXT NOT NULL CHECK (BTRIM(title) <> ''),
    email TEXT NOT NULL CHECK (BTRIM(email) <> ''),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS client_contacts_email_lower_uidx
    ON client_contacts (LOWER(email));

CREATE UNIQUE INDEX IF NOT EXISTS client_contacts_hubspot_record_id_uidx
    ON client_contacts (hubspot_record_id)
    WHERE hubspot_record_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS client_contacts_active_name_idx
    ON client_contacts (company_name, last_name, first_name)
    WHERE active = TRUE;

CREATE UNIQUE INDEX IF NOT EXISTS internal_contacts_email_lower_uidx
    ON internal_contacts (LOWER(email));

CREATE INDEX IF NOT EXISTS internal_contacts_active_name_idx
    ON internal_contacts (name)
    WHERE active = TRUE;

DROP TRIGGER IF EXISTS client_contacts_trim_text_trigger ON client_contacts;
CREATE TRIGGER client_contacts_trim_text_trigger
BEFORE INSERT OR UPDATE ON client_contacts
FOR EACH ROW
EXECUTE FUNCTION contact_management_trim_text();

DROP TRIGGER IF EXISTS internal_contacts_trim_text_trigger ON internal_contacts;
CREATE TRIGGER internal_contacts_trim_text_trigger
BEFORE INSERT OR UPDATE ON internal_contacts
FOR EACH ROW
EXECUTE FUNCTION contact_management_trim_text();

DROP TRIGGER IF EXISTS client_contacts_updated_at_trigger ON client_contacts;
CREATE TRIGGER client_contacts_updated_at_trigger
BEFORE UPDATE ON client_contacts
FOR EACH ROW
EXECUTE FUNCTION contact_management_set_updated_at();

DROP TRIGGER IF EXISTS internal_contacts_updated_at_trigger ON internal_contacts;
CREATE TRIGGER internal_contacts_updated_at_trigger
BEFORE UPDATE ON internal_contacts
FOR EACH ROW
EXECUTE FUNCTION contact_management_set_updated_at();

INSERT INTO schema_migrations (version)
VALUES ('contacts_v1')
ON CONFLICT (version) DO NOTHING;

