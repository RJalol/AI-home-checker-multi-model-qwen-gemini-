web: uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
worker: rq worker homework_grading webhook_delivery --with-scheduler
reconciler: python reconcile.py --loop 60

retention: sh -c 'while true; do python retention.py; sleep 86400; done'
