"""
Python + SQLAlchemy ORM 모델
DB 안의 asset_animations 테이블, 즉 캐릭터 Asset에 연결된 애니메이션 파일 정보를 저장하는 모델
캐릭터 Asset에 연결된 걷기/인사 같은 애니메이션 GLB 파일 정보와 Unity 실행 함수명을 저장하는 SQLAlchemy ORM 모델

"""


from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base

#asset_animations 테이블과 연결된 SQLAlchemy ORM 모델
#캐릭터 하나에 대해 생성된 애니메이션 정보를 저장
class AssetAnimation(Base):
    __tablename__ = "asset_animations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("assets.id"), 
        nullable=False)
    
    animation_key: Mapped[str] = mapped_column(String(50))   # walk / hello
    
    display_name: Mapped[str] = mapped_column(String(100))   # 걷기 / 인사_01
    
    file_url: Mapped[str | None] = mapped_column(String(500))
    
    unity_function: Mapped[str] = mapped_column(String(100)) # animation_walk / animation_Hello
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        server_default=func.now())

    asset: Mapped["Asset"] = relationship(
        "Asset", 
        back_populates="animations")
