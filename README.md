# Fault-simulation

+ [] open a github for this project and implement a very simple basic CPU baseline

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

## Record
* Ask questions (Ask_q) in the `simulation.cpp`
* `\n` and `std::endl`
	* Differnece: The only difference is that `std::endl` flushes the output buffer, and `\n` doesn't. If you don't want the buffer flushed frequently, use `\n`. If you do (for example, if you want to get all the output, and the program is unstable), use `std::endl`. [Ref](https://stackoverflow.com/questions/213907/stdendl-vs-n)
	* Though in some use cases like competitive programming, it is better to use “\n” when the problem is not interactive, as it saves time by not flushing the entire line and thus improves the time complexity of the program.[Ref](https://www.geeksforgeeks.org/endl-vs-n-in-cpp/)
* 