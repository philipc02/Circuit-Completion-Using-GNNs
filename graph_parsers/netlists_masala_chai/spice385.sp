* SPICE Netlist
* Vi input source
Vin 6 0 DC 0V

* Current Source
I1 8 0 DC 1/8 

* Transistors
Q12 3 6 0 QNPN
Q14 4 3 0 QNPN
Q15 2 4 5 QNPN

* Resistors
R12 3 0 R12_value
RL 5 0 RL_value

.model QNPN NPN (IS=1E-14 BF=100)

* Nodes
* Node 6: Positive terminal of Vi
* Node 0: Ground
* Node 3: Collector of Q12 and base of Q14
* Node 4: Collector of Q14 and base of Q15
* Node 5: Emitter of Q15 and connected to RL
* Node 2: Collector of Q15
* Node 8: Connected to Current Source

.end