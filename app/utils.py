import os
import mimetypes

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'tar', 'gz'}

def allowed_file(filename):
     return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
     file_type, encoding = mimetypes.guess_all_type(filename)
     if file_type :
          return file_type

     ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
     return f"application/{ext}" if ext else "application/octet-stream"
