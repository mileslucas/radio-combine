.PHONY: clean

clean:
	rm -rf *.log
	rm -rf data/*regrid*
	rm -rf scripts/*.pyc
	rm -rf tests/*.pyc
	rm -rf *.last
	rm -rf .nfs*
