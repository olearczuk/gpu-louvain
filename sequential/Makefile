CC          := g++
CFLAGS      := -O3 -Wall -c
LFLAGS      := -O3 -Wall
ALL         := seqlouvain


all : $(ALL)

seqlouvain : louvain_sequential.o utils.o modularity_optimisation.o community_aggregation.o
	$(CC) $(LFLAGS) -o $@ $^

clean :
	rm -f *.o $(ALL)
