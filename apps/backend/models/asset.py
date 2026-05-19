import uuid
from datetime import datetime
from sqlalchemy import String, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"asset_{uuid.uuid4().hex[:8]}")
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False)  # character / building / vehicle
    input_image_path: Mapped[str | None] = mapped_column(String(500))
    model_url: Mapped[str | None] = mapped_column(String(500))
    thumbnail_url: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/processing/completed/failed
    stage: Mapped[str] = mapped_column(String(30), default="waiting")  # waiting/generating/rigging/animating/downloading/ready
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    session: Mapped["Session"] = relationship("Session", back_populates="assets")
    animations: Mapped[list["AssetAnimation"]] = relationship("AssetAnimation", back_populates="asset", lazy="select")
    plaza_objects: Mapped[list["PlazaObject"]] = relationship("PlazaObject", back_populates="asset", lazy="select")
