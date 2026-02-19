* SPICE Netlist for the given circuit

Q1 2 1 9 NPN
Q2 3 2 9 NPN
Q3 4 2 2 NPN
Q4 5 2 2 NPN

RC1 8 3 100
RC1 7 3 100
RC2 4 5 100
RC2 8 5 100

I_Q1 1 9 DC 0.2mA
I_Q2 2 6 DC 0.4mA

V+ 8 0 DC 10V
V- 9 0 DC -10V

.end