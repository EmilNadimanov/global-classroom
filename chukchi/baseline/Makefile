all: model.dat output.tsv
	python3 ../evaluate.py ../data/test/test.tsv output.tsv

model.dat:
	python3 train.py ../data/train.tsv model.dat

output.tsv:
	python3 predict.py model.dat < ../data/test/test.tsv > output.tsv

clean:
	rm model.dat output.tsv

re: clean all