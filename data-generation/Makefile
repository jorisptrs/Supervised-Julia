
CC = g++
CFLAGS = -O2

fractals: Fractals.cpp
	$(CC) $(CFLAGS) $^ -o $@

run: fractals
	./fractals

clean:
	rm -f fractals