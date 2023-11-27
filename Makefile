# define the C compiler to use

CC = g++

SH = bash

RM = rm -f

# define any compile-time flags

CFLAGS = -Wall -Ofast -Wextra -std=c++1z # -D_GLIBCXX_DEBUG -D_FORTIFY_SOURCE=2 -pg --coverage -fprofile-abs-path#-fno-ipa-cp-clone -g -Og  #quando faccio il linking di fragprob -lgcov -pg

# define any directories containing header files other than /usr/include

WD=`pwd`

INCLUDES = -I${WD}#-I/usr/local/boost_1_72_0/ -L /usr/local/boost_1_72_0/stage/lib

#when perfecting the folder structure change this to ../include

HEADERS = lib

# define any libraries to link into executable:

#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 

#   option, something like (this will link in libmylib.so and libm.so:

LIBS = -lm -lpthread #-lboost_filesystem -lboost_system

VPATH=bin:src

ODIR=bin

# _DEPS = bookprob.hpp bookdkl.hpp supportlib.hpp paramOpt.hpp authorSplitter.hpp datatypes.hpp base_experiment.hpp

# DEPS = $(patsubst %,$(HEADERS)/%,$(_DEPS))

# _OBJ = bookprob.o bookdkl.o fragprob.o supportlib.o paramOpt.o authorSplitter.o base_experiment.o

# OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# CINP = $(wildcard *.c)

# COUT = $(patsubst %.c,$(ODIR)/%,$(CINP))

#

# The following part of the makefile is generic; it can be used to 

# build any executable just by changing the definitions above and by

# deleting dependencies appended to the file from 'make depend'

#

all:  $(ODIR)/libpmi.so

	@echo  All compiled


$(ODIR)/libpmi.so: $(ODIR)/pmi.o

	g++ $(CFLAGS) -shared -Wl,-soname,pmi.so -o $@ $^ $(INCLUDES) ${LIBS}

	@echo $@ shared object created

$(ODIR)/pmi.o: pmi.cpp

	g++ -c -fPIC $(CFLAGS) $^ -o $@ $(INCLUDES)

# this is a suffix replacement rule for building .o's from .c's

# it uses automatic variables $<: the name of the prerequisite of

# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 

# (see the gnu make manual section about automatic variables)

.PHONY: clean

clean:

	$(RM) $(ODIR)/* *~ core $(INCDIR)/*~ 

format:
	isort --profile black -l 100 mienc/
	black -l 100 mienc/