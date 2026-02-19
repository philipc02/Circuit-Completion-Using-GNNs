* SPICE Netlist for the given circuit

* NPN Transistor
Q1 3 6 2 NPN

* Resistors
RS 5 6 RS
RL 3 0 RL
RT 2 0 2RT

* Capacitors
CT_HALF 2 0 CT_HALF

* Current Source
IC 5 0 DC i_c

* Nodes
* 0: Ground
* 2: Emitter of NPN and lower node for 2RT, CT_HALF
* 3: Collector of NPN and upper node of RL
* 4: Base of NPN (connected node)
* 5: Positive terminal of i_c
* 6: Base of NPN and connected to RS

.model NPN NPN (IS=1e-15 BF=100 VAF=50 ISE=1e-15)

.end