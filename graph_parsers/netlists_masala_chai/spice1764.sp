* SPICE Netlist

V_N N 0 DC <value> ; Voltage source V_N connected between node N and ground
V_test M 0 DC <value> ; Voltage source V_test connected between node M and ground

A1 5 2 4 ; Amplifier block A1 with input at node 5, control node 2, and output node 4
K 3 3 ; Amplifier K with both input and output at node 3

* Node Mapping
* Node 0: Ground
* Node N: Node 1 (connected to positive terminal of V_N)
* Node M: Node 5 (connected to positive terminal of V_test)
* Node 2: Control input for A1
* Node 3: Output of K and input to A1
* Node 4: Output of A1 (connected to Y)

.end