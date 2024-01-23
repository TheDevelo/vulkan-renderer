CFLAGS = -std=c++20 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VKRenderer: main.cpp
	g++ $(CFLAGS) -o VKRenderer main.cpp $(LDFLAGS)

.PHONY: test clean

test: VKRenderer
	./VKRenderer

clean:
	rm -f VKRenderer
