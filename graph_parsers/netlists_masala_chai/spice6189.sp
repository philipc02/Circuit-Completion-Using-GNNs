plaintext
* SPICE netlist for the given schematic

* Voltage Sources
V1 1 0 DC 2.5
V2 3 0 DC -2.5
V3 5 0 DC v_s

* Current Sources
I1 2 3 DC 200u
I2 2 0 DC 300u
I3 4 3 DC 0.8m

* PMOS Transistors
Q1 2 2 1 PMOS L=1u W=120u
Q2 2 2 1 PMOS L=1u W=40u

* NMOS Transistors
Q3 6 5 3 NMOS L=1u W=20u
Q4 4 2 3 NMOS L=1u W=120u
Q5 0 4 3 NMOS L=1u W=20u

* Nodes
* 0: Ground
* 1: +2.5 V
* 2: Common node
* 3: -2.5 V
* 4: v_o
* 5: v_s
* 6: Junction of Q1, Q2, and I2

.END