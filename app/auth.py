import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from fastapi import HTTPException, Request

from . import config


def get_authenticated_user_id(request: Request) -> str:
    if not config.CLERK_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail="Clerk server authentication is not configured. Set CLERK_SECRET_KEY in backend/.env.",
        )

    clerk = Clerk(bearer_auth=config.CLERK_SECRET_KEY)
    request_state = clerk.authenticate_request(
        httpx.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
        ),
        AuthenticateRequestOptions(authorized_parties=[config.FRONTEND_ORIGIN]),
    )
    if not request_state.is_signed_in:
        raise HTTPException(status_code=401, detail="A valid Clerk session is required.")

    payload = request_state.payload or {}
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="The Clerk session has no user identity.")
    return user_id
