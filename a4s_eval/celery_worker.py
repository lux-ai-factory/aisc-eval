from a4s_eval.celery_tasks import celery_app



def main():
    celery_app.worker_main()

if __name__ == "__main__":
    main()
