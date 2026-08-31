"""Pydantic request/response contracts for the API.

These are the only shapes that cross the HTTP boundary — services and
core logic work with plain Python objects internally and only touch
these models at the edges (route handlers).
"""
