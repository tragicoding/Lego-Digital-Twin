from datetime import datetime
from sqlalchemy import String, Float, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base


class PlazaObject(Base):
    __tablename__ = "plaza_objects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    asset_id: Mapped[str | None] = mapped_column(String, ForeignKey("assets.id"))
    object_type: Mapped[str] = mapped_column(String(30))  # guide_npc / static_building / driveable_vehicle
    position_x: Mapped[float] = mapped_column(Float, default=0.0)
    position_y: Mapped[float] = mapped_column(Float, default=0.0)
    position_z: Mapped[float] = mapped_column(Float, default=0.0)
    rotation_y: Mapped[float] = mapped_column(Float, default=0.0)
    scale: Mapped[float] = mapped_column(Float, default=1.0)
    visible: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["Session"] = relationship("Session", back_populates="plaza_objects")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="plaza_objects")
