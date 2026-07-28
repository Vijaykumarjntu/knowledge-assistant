from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    status = Column(String, default="uploaded")
    category = Column(String, nullable=True)
    metadata1 = Column(Text, nullable=True)

    chunks = relationship("Chunk", cascade="all, delete-orphan", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    vector_id = Column(Integer, unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

    document = relationship("Document", back_populates="chunks")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=True)
    query = Column(Text, nullable=False)
    response_type = Column(String, nullable=False)
    referenced_docs = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
