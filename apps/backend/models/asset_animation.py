from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base


class AssetAnimation(Base):
    __tablename__ = "asset_animations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String, ForeignKey("assets.id"), nullable=False)
    animation_key: Mapped[str] = mapped_column(String(50))   # walk / hello
    display_name: Mapped[str] = mapped_column(String(100))   # 걷기 / 인사_01
    file_url: Mapped[str | None] = mapped_column(String(500))
    unity_function: Mapped[str] = mapped_column(String(100)) # animation_walk / animation_Hello
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    asset: Mapped["Asset"] = relationship("Asset", back_populates="animations")
