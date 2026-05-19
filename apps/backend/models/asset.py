"""
Python + SQLAlchemy ORM 모델
DB 안의 assets 테이블, 즉 업로드된  레고 이미지/3D 모델 정보를 표현하는 클래스
업로드된 레고 이미지 하나를 DB에서 관리하기 위한 SQLAlchemy ORM 모델
"""

'''
uuid: 고유한 asset ID 생성용
datetime: 생성/수정 시간 타입 지정용
String, Integer, DateTime: DB 컬럼 타입
ForeignKey: 다른 테이블과 연결
func.now(): DB 현재 시간 사용
Mapped, mapped_column: SQLAlchemy 2.x 스타일 컬럼 선언
relationship: 다른 모델과 객체 관계 연결'''
import uuid
from datetime import datetime
from sqlalchemy import String, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.database import Base

#Base를 상속받았기 때문에 SQLAlchemy가 이 클래스를 테이블로 인식
class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"asset_{uuid.uuid4().hex[:8]}")
    
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.id"), 
        nullable=False)
    
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False)  # character / building / vehicle
    
    input_image_path: Mapped[str | None] = mapped_column(String(500))

    model_url: Mapped[str | None] = mapped_column(String(500))

    thumbnail_url: Mapped[str | None] = mapped_column(String(500))

    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/processing/completed/failed
    
    stage: Mapped[str] = mapped_column(String(30), default="waiting")  # waiting/generating/rigging/animating/downloading/ready
    
    progress: Mapped[int] = mapped_column(Integer, default=0)

    error_message: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        server_default=func.now())
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        server_default=func.now(), 
        onupdate=func.now())

    session: Mapped["Session"] = relationship(
        "Session", back_populates="assets")
    
    animations: Mapped[list["AssetAnimation"]] = relationship(
        "AssetAnimation", back_populates="asset", 
        lazy="select")
    plaza_objects: Mapped[list["PlazaObject"]] = relationship(
        "PlazaObject", back_populates="asset", 
        lazy="select")
