CC          := g++
CFLAGS      := -O3 -Wall -c
LFLAGS      := -O3 -Wall
ALL         := seqlouvain


all : $(ALL)

seqlouvain : sequential/louvain_sequential.o sequential/utils.o sequential/modularity_optimisation.o sequential/community_aggregation.o
	$(CC) $(LFLAGS) -o $@ $^

clean :
	rm -f sequential/*.o $(ALL)
