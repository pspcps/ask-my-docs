"""Run the persistent indexing queue in a separate process."""

import logging
import time

from service import DocumentService


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = DocumentService()
    logging.info("Watching document jobs in %s", service.path)
    try:
        while True:
            if not service.process_next():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
