all:
	make -C pi
	make -C hello_world
	make -C abzahlungsdarlehen

clean:
	rm -rfv bin lib