

test:
	$(MAKE) -C sensor/vision/test all
	$(MAKE) -C perception/vision/ml

model:
	$(MAKE) -C structure all

clean:
	$(MAKE) -C sensor/vision/test clean
	$(MAKE) -C perception/vision/ml clean
	$(MAKE) -C structure clean

.PHONY: test model clean