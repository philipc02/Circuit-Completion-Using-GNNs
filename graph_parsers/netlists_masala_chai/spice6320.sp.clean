spice
* SPICE Netlist for the circuit

V_R 1 0 DC <VR_value>       ; Define the voltage for VR later
R1 3 0 907
R2 3 4 4.98k
R3 4 2 6.1k
Q1 3 2 4 NPN                 ; Assume a generic NPN model
D1 3 3 D_model               ; Define the diode model later
D2 3 4 D_model               ; Define the diode model later

.model NPN NPN(Is=1e-15)     ; Example NPN model
.model D_model D(Is=1e-15)   ; Example diode model

.END