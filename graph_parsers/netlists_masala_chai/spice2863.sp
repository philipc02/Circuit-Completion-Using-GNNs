plaintext
* SPICE Netlist
V1 3 8 DC Vn1       ; Voltage source Vn1 connected between node 3 and ground (8)

M1 7 3 4 4 NMOS     ; NMOS M1 with Drain: node 7 (VDD), Gate: node 3, Source: node 4
M2 4 3 6 6 NMOS     ; NMOS M2 with Drain: node 4, Gate: node 3, Source: ground

I2 4 6 DC In2       ; Current source In2 connected between node 4 and ground (6)

* Nodes:
* 2: V²ₙ,out
* 3: Gate of both M1 and M2
* 4: Source of M1 and Drain of M2
* 6: Ground for current source I²ₙ₂
* 7: VDD
* 8: Ground for voltage source Vₙ₁

.end