plaintext
* SPICE netlist for the schematic
V1 6 8 DC <VALUE_OF_VE>    ; Voltage source ve
R1 6 8 <VALUE_OF_RE>       ; Resistor re
C1 6 10 <VALUE_OF_CPI>     ; Capacitor Cpi
C2 5 2 <VALUE_OF_CMU>      ; Capacitor Cmu
G1 7 5 V1 <VALUE_OF_GM>    ; Dependent current source gm*vpi

* Nodes
* 6: Positive terminal of ve and connection to re
* 8: Negative terminal of ve and ground connection for re
* 9: Internal connection within re
* 3: Intermediate connection between ve and Cpi
* 10: Connection for Cpi to Node b
* 5: Node connection for Cmu and output of G1
* 2: Node connection for the side opposite Cmu