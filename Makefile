

test:
	$(MAKE) -C sensor/vision/test all
	$(MAKE) -C perception/vision/ml
	$(MAKE) -C algorithm/cl/test test

model:
	$(MAKE) -C structure all

clean:
	$(MAKE) -C sensor/vision/test clean
	$(MAKE) -C perception/vision/ml clean
	$(MAKE) -C structure clean
	$(MAKE) -C system/test/module clean
	$(MAKE) -C algorithm/cl clean
	$(MAKE) -C algorithm/cl/test clean

.PHONY: test model clean