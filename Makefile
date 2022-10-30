.PHONY: clean md me


all:clean

clean:
	-rm -rf */__pycache__/
	-rm -r output/*
	-rm -r data/*

md: # make data
	-make clean
	python ./app/render_data.py

gen:
	-make clean
	-python ./app/gen_data.py
