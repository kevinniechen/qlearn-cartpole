SHELL = /bin/sh

define PROJECT_HELP_MSG

Usage:
	make help			show this message
	make clean 			remove intermediate files
	make run			quickstart classification

endef
export PROJECT_HELP_MSG

help:
	echo $$PROJECT_HELP_MSG | less

CLEAN_EXT = *.pyc
CLEAN_DIR = data/*

clean:
	rm -rf ${CLEANUP}
	rm -rf ${CLEAN_DIR}

run:
	python3 -W src/cartpole.py

.PHONY: help clean