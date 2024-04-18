
OUT=test
OU=$(OUT)
PPP=fun
srcfiles = $(shell python3 -m pybind11 --includes)#立即执行shell命令，得到其文本结果

a.txt b.txt: #make a.txt 生成a.txt
	touch $@



default:alost
	echo end

%lost:%.c
	cat $*.c

output:
	$(CC) -o test0 a.o b.o

var-lost:
	export foo=bar; \
	echo "foo=[$$foo]"

fun:
	@echo $(OU)
	@echo $(HOME)
test:a.o b.o          #test是目标，它依赖于a.o b.o文件，一旦a.o或者b.o比                              #test新的时候，就需要执行下面的命令，重新生成test可执行程序。
	gcc -o test a.o b.o         
a.o:a.c                  #a.o依赖于a.c，当a.c更加新的话，执行下面的命令来生成a.o
	gcc -c -o a.o a.c
b.o:b.c                 #b.o依赖于b.c,当b.c更加新的话，执行下面的命令，来生成b.o
	gcc -c -o b.o b.c
