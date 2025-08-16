
all: bottom_shell.stl top_shell.stl


%.stl: %.scad
	openscad -o $@ $<

clean:
	rm -f *.stl

.PHONY: all clean