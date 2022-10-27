.PHONY: clean md me


all:clean
# 清一下造出来的数据
clean:
	-rm -rf */__pycache__/
	-rm -r output/*
	-rm -r data/*

md: # make data
	-make clean
	python ./app/render_data.py
