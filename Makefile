CC = hipcc
SOURCES = main.hip gpu.hip kernels.hip
OBJECTS = $(patsubst %.hip,build/%.o,$(SOURCES))
TARGET = build/test


all: build $(TARGET)

build/%.o: %.hip
	$(CC) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) -g -o $(TARGET) $(OBJECTS)

build: 
	mkdir -p ./build

clean: 
	rm -rf ./build


run: 
	./$(TARGET)
