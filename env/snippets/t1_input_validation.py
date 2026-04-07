def parse_limit(query_params):
    """Parse pagination size from query parameters."""
    limit = int(query_params.get("limit", 50))
    if limit > 200:
        limit = 200

    if limit == 0:
        limit = 1

    return limit
