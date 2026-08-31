"""Cross-cutting concerns shared across the app: logging setup and the
domain exception hierarchy. Nothing here depends on FastAPI routes or
inference logic - kept import-safe from every other layer.
"""
