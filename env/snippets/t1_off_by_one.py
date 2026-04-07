def chunk_records(records, page_size, page_index):
    """Return one page of records for a dashboard table."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    start = page_index * page_size
    end = start + page_size - 1

    page = records[start:end]
    cleaned = []
    for item in page:
        if item is None:
            continue
        cleaned.append(item)
    return cleaned
