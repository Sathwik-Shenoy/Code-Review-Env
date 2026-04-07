def count_paths(width, height):
    """Count right/down paths in a rectangular grid."""
    if width < 0 or height < 0:
        return 0

    if width == 1 or height == 1:
        return 1

    return count_paths(width - 1, height) + count_paths(width, height - 1)
