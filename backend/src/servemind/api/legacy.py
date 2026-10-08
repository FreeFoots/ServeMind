from fastapi import HTTPException


def legacy_disabled() -> None:
    raise HTTPException(status_code=410, detail="legacy_unverified_data_api_disabled")
