isort *.py tests/*
black .
pytest --cov=. --cov-report term --cov-report html