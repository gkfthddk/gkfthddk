toim : toim.cxx
	g++ -std=c++0x toim.cxx -o toim `root-config --cflags --libs`
dup : dup.cxx
	g++ -std=c++0x dup.cxx -o dup `root-config --cflags --libs`
