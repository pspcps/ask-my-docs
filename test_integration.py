"""Exercise the HTTP interface and independent worker against the same database."""

import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from api import create_app
from service import DocumentService
from workspace import ROOT


class FixtureEngine:
    def __init__(self):
        self.digest = "fixture-embedding-v1"
        self.response = None

    def metadata(self):
        return {
            "embedding_model": "fixture-embed", "embedding_digest": self.digest,
            "chat_model": "fixture-chat", "chat_digest": "fixture-chat-v1",
        }

    def embed(self, texts):
        return np.array([[0, 1] if "unrelated" in text.lower() else [1, 0] for text in texts])

    def generate(self, question, evidence):
        if self.response is not None:
            return self.response
        for number in ("18", "22"):
            if f"{number} working days" in evidence[0]["text"]:
                return {"answer": f"{number} working days.", "citations": [1]}
        return {"answer": "", "citations": []}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "test.sqlite3"
        self.engine = FixtureEngine()
        self.service = DocumentService(self.path, self.engine)
        self.worker = DocumentService(self.path, self.engine)
        self.api = create_app(self.service).test_client()

    def upload(self, version=1, document_id=None, data=None):
        if data is None:
            data = (ROOT / "examples" / f"field-handbook-v{version}.pdf").read_bytes()
        endpoint = f"/documents/{document_id}" if document_id else "/documents"
        call = self.api.put if document_id else self.api.post
        return call(endpoint, data={"file": (io.BytesIO(data), "handbook.pdf")})

    def ask(self, question="How much annual leave?", **kwargs):
        return self.api.post("/ask", json={"question": question, **kwargs})

    def test_pdf_update_delete_and_shared_citations(self):
        response = self.upload()
        self.assertEqual(response.status_code, 202)
        queued = response.json
        document_id = queued["document_id"]
        self.assertEqual(self.api.get(f"/jobs/{queued['id']}").json["status"], "queued")
        duplicate = self.upload()
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.json["id"], queued["id"])
        self.assertTrue(self.worker.process_next())
        self.assertFalse(self.worker.process_next())

        first = self.ask().json
        direct = self.worker.ask("How much annual leave?")
        self.assertEqual(first["answer"], direct["answer"])
        self.assertEqual(first["citations"], direct["citations"])
        citation, = first["citations"]
        self.assertEqual(citation["page"], 1)
        self.assertEqual(citation["document_name"], "handbook.pdf")
        self.assertIn("18 working days", citation["excerpt"])
        self.assertEqual(self.ask(document_ids=[]).json["reason"], "insufficient_retrieval")

        second = self.upload(2, document_id).json
        self.assertEqual(second["version"], 2)
        self.assertEqual(self.ask().json["answer"], "18 working days.")
        self.worker.process_next()
        updated = self.ask().json
        self.assertEqual(updated["answer"], "22 working days.")
        self.assertEqual(updated["citations"][0]["version"], 2)
        record, = self.api.get("/documents").json["documents"]
        self.assertNotEqual(record["active"]["content_hash"], queued["content_hash"])
        self.assertEqual(record["chunk_ids"], [updated["citations"][0]["chunk_id"]])
        with self.worker.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)

        self.assertEqual(self.api.delete(f"/documents/{document_id}").status_code, 204)
        self.assertEqual(self.api.get(f"/jobs/{second['id']}").status_code, 404)
        self.assertEqual(self.api.post("/retrieve", json={"question": "annual leave"}).json["passages"], [])
        self.assertTrue(self.ask().json["abstained"])
        with self.worker.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)

    def test_failure_retry_and_embedding_rebuild(self):
        initial = self.upload().json
        self.worker.process_next()
        document_id = initial["document_id"]
        update = self.upload(2, document_id).json
        with patch.object(self.engine, "embed", side_effect=ValueError("simulated outage")):
            self.worker.process_next()
        failed = self.api.get(f"/jobs/{update['id']}").json
        self.assertEqual(failed["status"], "failed")
        self.assertIn("simulated outage", failed["error"])
        self.assertEqual(self.ask().json["answer"], "18 working days.")
        self.assertEqual(self.api.post(f"/jobs/{update['id']}/retry").status_code, 202)
        self.worker.process_next()
        self.assertEqual(self.api.get(f"/jobs/{update['id']}").json["attempts"], 2)
        self.assertEqual(self.ask().json["answer"], "22 working days.")

        self.engine.digest = "new-digest"
        self.assertEqual(self.ask().status_code, 409)
        self.assertEqual(self.api.post(f"/documents/{document_id}/rebuild").status_code, 202)
        self.worker.process_next()
        self.assertEqual(self.ask().json["citations"][0]["version"], 3)

    def test_invalid_citations_and_insufficient_evidence(self):
        self.upload()
        self.worker.process_next()
        self.assertEqual(self.ask("Unrelated subject").json["reason"], "insufficient_retrieval")
        self.engine.response = {"answer": "Trust me", "citations": [999]}
        self.assertEqual(self.ask().json["reason"], "insufficient_support")
        self.assertEqual(self.ask().json["citations"], [])
        self.assertEqual(self.ask(limit=0).status_code, 400)
        self.assertEqual(self.ask(document_ids="bad").status_code, 400)
        self.assertEqual(self.api.post("/documents").status_code, 400)

    def test_concurrent_duplicate_uploads_share_one_job(self):
        data = (ROOT / "examples" / "field-handbook-v1.pdf").read_bytes()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = list(pool.map(lambda _: self.service.enqueue("handbook.pdf", data), range(2)))
        self.assertEqual(jobs[0]["id"], jobs[1]["id"])
        self.assertEqual(sum(job["duplicate"] for job in jobs), 1)
        self.assertEqual(len(self.service.documents()), 1)

    def test_delete_during_indexing_cannot_resurrect_document(self):
        job = self.upload().json
        original = self.engine.embed

        def delete_and_embed(texts):
            self.service.delete(job["document_id"])
            return original(texts)

        with patch.object(self.engine, "embed", side_effect=delete_and_embed):
            self.worker.process_next()
        self.assertEqual(self.service.documents(), [])
        self.assertEqual(self.api.get(f"/jobs/{job['id']}").status_code, 404)

    def test_expired_job_visible_as_failed_and_retryable(self):
        job = self.upload().json
        self.worker._claim()
        with self.worker.connection(write=True) as db:
            db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job["id"],))
        self.assertFalse(self.worker.process_next())
        self.assertEqual(self.service.job(job["id"])["status"], "failed")
        self.service.retry(job["id"])
        self.worker.process_next()
        self.assertEqual(self.service.job(job["id"])["status"], "ready")


if __name__ == "__main__":
    unittest.main()
