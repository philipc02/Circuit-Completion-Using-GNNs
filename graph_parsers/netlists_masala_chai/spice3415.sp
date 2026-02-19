spice
* Voltage Source
V1 1 8 DC 0

* Resistors
R_RT 1 e  R_T
R_RG e 4  R_G
R_Re 4 2  R_e
R_RC 3 2  R_C
R_RL 3 2  R_L

* Dependent Current Source
G_IC b 3 e 0 gm

* Connections (where b, c, etc. correspond to points from the image)
* Nodes in SPICE netlist should match nodes from labeled image
* Assuming e, v_b are internal nodes corresponding to the circuit operation

* End of Netlist