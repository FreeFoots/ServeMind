from __future__ import annotations

from typing import Literal
from decimal import Decimal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from servemind.service.commerce_store import CommerceStore
from servemind.config.settings import APP_ENV, DEMO_ACCOUNT_SWITCH


router = APIRouter(prefix="/commerce", tags=["commerce"])


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=60)
    role: Literal["buyer", "merchant"]

    @field_validator("username", "display_name", mode="before")
    @classmethod
    def strip_public_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class DemoSelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1, max_length=100)


class ProductCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    sku_id: str | None = Field(default=None, max_length=80)
    category: str = Field(default="其他商品", min_length=1, max_length=40)
    price: Decimal | None = Field(default=None, gt=0, max_digits=10, decimal_places=2)


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)


class MessageCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    content: str = Field(min_length=1, max_length=2000)
    client_message_id: str | None = Field(default=None, max_length=128)


class HandoffUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["merchant_processing", "resolved"]


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: Literal["helpful", "not_helpful", "incorrect"]
    note: str = Field(default="", max_length=500)


def _store(request: Request) -> CommerceStore:
    return request.app.state.commerce_store


def _local_demo_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    if not (APP_ENV == "development" and DEMO_ACCOUNT_SWITCH
            and host in {"127.0.0.1", "::1", "testclient"}):
        raise HTTPException(status_code=404, detail="not_found")


def _account(request: Request, authorization: str | None = Header(default=None)) -> dict[str, str]:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="authentication_required")
    account = _store(request).authenticate(token)
    if account is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    return account


@router.post("/accounts/register")
def register(payload: RegisterRequest, request: Request):
    return _store(request).register(**payload.model_dump())


@router.post("/accounts/login")
def login(payload: LoginRequest, request: Request):
    return _store(request).login(**payload.model_dump())


@router.get("/demo/accounts", dependencies=[Depends(_local_demo_only)])
def demo_accounts(request: Request):
    return {"items": _store(request).list_demo_accounts()}


@router.post("/demo/select", dependencies=[Depends(_local_demo_only)])
def demo_select(payload: DemoSelectRequest, request: Request):
    return _store(request).select_demo_account(payload.account_id)


@router.get("/me")
def me(account: dict[str, str] = Depends(_account)):
    return {"account": account}


@router.get("/products")
def products(request: Request, account: dict[str, str] = Depends(_account)):
    return {"items": _store(request).list_products()}


@router.get("/purchases")
def purchases(request: Request, account: dict[str, str] = Depends(_account)):
    return {"items": _store(request).list_purchases(account=account)}


@router.post("/products")
def create_product(payload: ProductCreate, request: Request, account: dict[str, str] = Depends(_account)):
    return _store(request).create_product(account=account, **payload.model_dump())


@router.get("/conversations")
def conversations(request: Request, account: dict[str, str] = Depends(_account)):
    return {"items": _store(request).list_conversations(account=account)}


@router.post("/conversations")
def create_conversation(payload: ConversationCreate, request: Request, account: dict[str, str] = Depends(_account)):
    return _store(request).create_conversation(account=account, product_id=payload.product_id)


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, request: Request, account: dict[str, str] = Depends(_account)):
    return _store(request).get_conversation(account=account, conversation_id=conversation_id)


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    payload: MessageCreate,
    request: Request,
    account: dict[str, str] = Depends(_account),
):
    return _store(request).send_message(
        account=account,
        conversation_id=conversation_id,
        content=payload.content,
        client_message_id=payload.client_message_id,
        support=request.app.state.commerce_support,
    )


@router.post("/conversations/{conversation_id}/handoff")
def handoff_update(conversation_id: str, payload: HandoffUpdate, request: Request,
                   account: dict = Depends(_account)):
    return _store(request).update_handoff(account=account, conversation_id=conversation_id, state=payload.state)


@router.post("/conversations/{conversation_id}/messages/{message_id}/feedback")
def feedback(conversation_id: str, message_id: str, payload: FeedbackCreate, request: Request,
             account: dict = Depends(_account)):
    return _store(request).record_feedback(account=account, conversation_id=conversation_id,
                                          message_id=message_id, **payload.model_dump())


@router.get("/diagnostics", dependencies=[Depends(_local_demo_only)])
def diagnostics(request: Request, account: dict = Depends(_account)):
    return _store(request).diagnostics()
