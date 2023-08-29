export PYTHONPATH=:~/Projects/CurriculAI
isort --profile black *.py tests/*
black .
pytest -n 10 --cov=. --cov-report term --cov-report html
