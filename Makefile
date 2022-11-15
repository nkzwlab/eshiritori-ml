filename ?=

init:
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up -d

run:
	docker-compose -f docker-compose.yml exec eshiritori-ml python -B ${filename}

down:
	docker-compose -f docker-compose.yml down

install:
	docker-compose -f docker-compose.yml exec eshiritori-ml pip install -r requirements.txt