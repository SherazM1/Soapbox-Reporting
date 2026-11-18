# Photography contacts and drafts

Photography contacts, quote drafts, and version history use the existing cloud
PostgreSQL database. Configure the same `DATABASE_URL` for every app installation,
using Streamlit secrets or the environment. Streamlit secrets take precedence.
The PostgreSQL driver is included in `requirements.txt`.

If the connection is missing or fails, the app reports an error. It does not create
a local database or switch to a different data source. Starting the app does not
initialize the cloud schema or import local records.

**Create Draft** starts a blank quote. Select a saved draft and click **Edit Draft**
to load it. **Save Draft** saves a new draft or replaces the current editable
contents of the active draft. Names are generated from the title, company, and date.

Current contents are stored in `quote_draft_working_copies` in the same cloud database.
That table is created if needed by the first create, edit, or save operation; the
database account needs permission to create it. Its additive definition is in
`draft_working_copies.sql`. No local database is created.

Existing drafts initially load their latest historical version until an editable
copy is saved. Saving uses the working-copy table only. Existing draft records,
version numbers, payloads, timestamps, and version notes are not modified or deleted.
New drafts do not create version records. Version and deletion controls are absent
from the UI. Legacy repository functions remain for compatibility but are not used
by this workflow.

The existing cloud database already has its data and schema. Do not run the database
initialization or contact import scripts as part of this change. Those tools remain
available for explicit administrative setup and imports.
