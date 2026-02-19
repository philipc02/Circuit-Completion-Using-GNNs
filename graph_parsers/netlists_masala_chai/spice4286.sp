plaintext
* NMOS Transistors: M<name> <drain> <gate> <source> <body> ModelName L=<value> W=<value>
* Current Source: I<name> <positive_terminal> <negative_terminal> DC <value>

* Transistors
M1 0 2 2 2 NMOS
M2 2 3 5 5 NMOS
M3 3 3 5 5 NMOS
M4 4 4 5 5 NMOS

* Current Source
IREF 1 2 DC <IREF_value>

* Voltage Source
V+ 0 1 DC 1.8V
V- 5 0 DC -1.8V

* Load currents
IO1 0 3 DC 0.1mA
IO2 0 4 DC 0.2mA
IO3 0 5 DC 0.4mA

.model NMOS NMOS LEVEL=1