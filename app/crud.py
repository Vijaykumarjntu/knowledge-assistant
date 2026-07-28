from sqlalchemy.orm import Session

from .models import Document, Chunk, Interaction


def get_document_by_id(db: Session, document_id: str):
    return db.query(Document).filter(Document.document_id == document_id).first()


def list_documents(db: Session):
    return db.query(Document).order_by(Document.upload_timestamp.desc()).all()


def create_document(db: Session, document_id: str, name: str, file_path: str):
    document = Document(document_id=document_id, name=name, file_path=file_path, status="uploaded")
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def delete_document(db: Session, document: Document):
    db.delete(document)
    db.commit()


def delete_chunks_by_document(db: Session, document_pk: int):
    db.query(Chunk).filter(Chunk.document_id == document_pk).delete()
    db.commit()


def add_interaction(db: Session, session_id: str, query: str, response_type: str, referenced_docs: str = None):
    interaction = Interaction(
        session_id=session_id,
        query=query,
        response_type=response_type,
        referenced_docs=referenced_docs,
    )
    db.add(interaction)
    db.commit()
    return interaction


def list_interactions(db: Session):
    return db.query(Interaction).order_by(Interaction.timestamp.desc()).all()
