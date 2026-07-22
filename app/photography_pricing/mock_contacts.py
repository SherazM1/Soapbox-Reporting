MOCK_INTERNAL_CONTACTS = (
    {
        "id": "ashley-watson",
        "name": "Ashley Watson",
        "title": "Photography Producer",
        "email": "ashley.watson@soapbox.com",
    },
    {
        "id": "morgan-lee",
        "name": "Morgan Lee",
        "title": "Creative Operations Manager",
        "email": "morgan.lee@soapbox.com",
    },
    {
        "id": "jordan-reyes",
        "name": "Jordan Reyes",
        "title": "Account Lead",
        "email": "jordan.reyes@soapbox.com",
    },
)


def contact_options() -> tuple[dict[str, str], ...]:
    return MOCK_INTERNAL_CONTACTS
