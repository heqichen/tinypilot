

test:
	$(MAKE) -C sensor/vision/test all
	$(MAKE) -C perception/vision/ml
clean:
	$(MAKE) -C sensor/vision/test clean
	$(MAKE) -C sensor/vision/test clean

.PHONY: clean