# Fault-simulation

+[] open a github for this project and implement a very simple basic CPU baseline

```
~$ cd yhsim
~$ mkdir build
~$ cd build
~$ cmake ../
~$ ./bin/yhsim ../benchmarks/simple.ckt ../benchmarks/simple.flst ../benchmarks/simple.ptn

```

Inside simple.ckt (c7552.ckt), you can define the fault simulation benchmark format that you feel comfortable with

As a starting point, you can create a simple.ckt/simple.flst/simple.ptn benchmark that represents the example of 李建模's slide.

* files
	- .ckt: circuit file
	- .flst: fault list file
	- .ptn: pattern list file

---

## todo
* revise code better, with better comment
* pattern/fault have mapping index problem need to fix
* make a PPT 
