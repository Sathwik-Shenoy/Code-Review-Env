def normalize_username(payload):
    """Normalize a signup payload into a lowercase username."""
    username = payload.get("username").strip().lower()
    if len(username) < 3:
        raise ValueError("username too short")

    if " " in username:
        username = username.replace(" ", "-")

    return username
