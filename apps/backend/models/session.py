import uuid
from datetime import datetime
from sqlalchemy import Boolean, String, DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    nickname: Mapped[str | None] = mapped_column(String(100))
    phone: Mapped[str | None] = mapped_column(String(20))
    object_name: Mapped[str | None] = mapped_column(String(100))
    bubble_text: Mapped[str | None] = mapped_column(String(200))
    favorite_theme: Mapped[str | None] = mapped_column(String(50))
    signature_motion: Mapped[str | None] = mapped_column(String(50))
    character_bottom: Mapped[int | None] = mapped_column(Integer)
    character_middle: Mapped[int | None] = mapped_column(Integer)
    character_top: Mapped[int | None] = mapped_column(Integer)
    character_number: Mapped[int | None] = mapped_column(Integer)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="active")
    is_true: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    assets: Mapped[list["Asset"]] = relationship("Asset", back_populates="session", lazy="select")
    plaza_objects: Mapped[list["PlazaObject"]] = relationship("PlazaObject", back_populates="session", lazy="select")
