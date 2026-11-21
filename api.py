"""Local HTTP interface for the shared document service."""

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from service import MAX_BYTES, DocumentService, ServiceError


def create_app(service=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 64 * 1024
    service = service or DocumentService()

    @app.errorhandler(ServiceError)
    def service_error(error):
        return jsonify(error=str(error)), error.status

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    def upload(document_id=None):
        file = request.files.get("file")
        if file is None:
            raise ServiceError("Attach a multipart file using the file field.")
        job = service.enqueue(file.filename, file.read(MAX_BYTES + 1), document_id)
        return jsonify(job), 200 if job["duplicate"] else 202

    def query():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ServiceError("Send a JSON object with a question.")
        return payload.get("question"), payload.get("document_ids"), payload.get("limit", 4)

    @app.get("/health")
    def health():
        return jsonify(status="ok", storage="sqlite")

    @app.get("/documents")
    def list_documents():
        return jsonify(documents=service.documents())

    @app.post("/documents")
    def create_document():
        return upload()

    @app.put("/documents/<document_id>")
    def update_document(document_id):
        return upload(document_id)

    @app.delete("/documents/<document_id>")
    def delete_document(document_id):
        service.delete(document_id)
        return "", 204

    @app.post("/documents/<document_id>/rebuild")
    def rebuild_document(document_id):
        return jsonify(service.rebuild(document_id)), 202

    @app.get("/jobs/<job_id>")
    def job_status(job_id):
        return jsonify(service.job(job_id))

    @app.post("/jobs/<job_id>/retry")
    def retry_job(job_id):
        return jsonify(service.retry(job_id)), 202

    @app.post("/retrieve")
    def retrieve():
        return jsonify(passages=service.retrieve(*query()))

    @app.post("/ask")
    def ask():
        return jsonify(service.ask(*query()))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000)
