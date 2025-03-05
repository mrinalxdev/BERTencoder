from app import db
from datetime import datetime, timezone
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class File(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id' : self.id,
            'filename' : self.filename,
            'original_filename' : self.original_filename,
            'file_type' : self.file_type,
            'file_size' : self.file_size,
            'created_at' : self.created_at.isoformat(),
            'updated_at' : self.updated_at.isoformat()
        }

    def __repr__(self):
        return f'<File {self.original_filename}>'
