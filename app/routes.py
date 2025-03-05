from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
import os
from app.models import File
from app import db
from app.utils import allowed_file, get_file_type

main_bp = Blueprint('main', __name__)

@main_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status' : 'ok'}), 200

@main_bp.route('/files', methods=['GET'])
def get_files():
    files = File.query.all()
    return jsonify([file.to_dict() for file in files]), 200

@main_bp.route('/files/<file_id>', methods=['GET'])
def get_file(file_id):
    file = File.query.get_or_404(file_id)
    return jsonify(file.to_dict()), 200

@main_bp.route('/files', method=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error' : 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error' : 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error' : 'File type not allowed'}), 400

    original_filename = file.filename
    filename = secure_filename(original_filename)
    filename = f"{os.path.splitext(filename)[0]}_{int(datetime.now(timezone.utc()).timestamp())}"

    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    file_size = os.path.getsize(file_path)
    file_type = get_file_type(filename)

    new_file = File (
        filename=filename,
        original_filename=original_filename,
        file_type=file_type,
        file_size=file_size
    )

    db.session.add(new_file)
    db.session.commit()

    return jsonify(new_file.to_dict()), 201

@main_bp.route('/file/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    file = File.query.get_or_404(file_id)

    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
    if os.path.exists
