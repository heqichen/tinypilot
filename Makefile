

test:
	$(MAKE) -C sensor/vision/test all
clean:
	$(MAKE) -C sensor/vision/test clean

.PHONY: clean