export PYTHONPATH=:~/Projects/CurriculAI
isort --profile black *.py tests/*
black .
pytest -n auto --cov=. --cov-report term --cov-report html
