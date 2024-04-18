CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python3 -m pybind11 --includes)#立即执行shell命令，得到其文本结果
LIBNAME = helpers
LIBEXT = $(shell python3-config --extension-suffix)

default: $(LIBNAME)$(LIBEXT)#这是组合在一起的依赖，比如helpersxxxxx，make会去找这个依赖，发现有%xxxxx，通配符，%=helpers

%$(LIBEXT): %.cpp#$<指第一个依赖，$@指%$(LIBEXT)的值，这个makefile就是编译生成了helpers.cpython-310-x86_64-linux-gnu.so这个c++可执行文件
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
