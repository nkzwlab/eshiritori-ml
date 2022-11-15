filename ?=

init:
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up -d

run:
	docker-compose -f docker-compose.yml exec artalk python -B ${filename}

down:
	docker-compose -f docker-compose.yml down

install:
	docker-compose -f docker-compose.yml exec artalk pip install -r requirements.txt