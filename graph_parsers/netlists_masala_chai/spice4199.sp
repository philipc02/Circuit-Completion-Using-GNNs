plaintext
* SPICE Netlist for the given circuit

V1 4 0 DC <value of v_I>      * Voltage source with node 4 to ground

R1 4 2 <value of R1>          * Resistor R1 between node 4 and node 2
R2 2 3 <value of R2>          * Resistor R2 between node 2 and node 3
R3 4 2 <value of R3>          * Resistor R3 between node 4 and node 2
R4 2 0 <value of R4>          * Resistor R4 between node 2 and ground

* Op-Amp model
* "+" input at node 2 and "-" input also at node 2
* Output at node 3
XOP 2 2 3 OPAMP              * Generic op-amp model 

.end