plaintext
* SPICE Netlist

* Voltage Source
Vs 8 0 DC <value>

* Capacitors
C1 8 3 <value_C1>
C2 3 4 <value_C2>

* Switches
S1 7 2 <model_S1> <control_node> 0
S2 2 0 <model_S2> <control_node> 0
S3 3 0 <model_S3> <control_node> 0
S4 3 4 <model_S4> <control_node> 0
S5 4 5 <model_S5> <control_node> 0

* Operational Amplifier
.subckt opamp noninverting_input inverting_input output
R1 noninverting_input output <value_R>
C1 noninverting_input output <value_C>
.ends opamp

XOPAMP 3 4 5 opamp

* Control Voltages for switches
Vphi1 <control_node> 0 DC <value_phi1>
Vphi2 <control_node> 0 DC <value_phi2>

.end