plaintext
* Voltage Source
Vs 8 0 DC <value_of_Vs>

* Basic Amplifier Section
R1 7 2 <value_of_h11>
G1 2 0 5 6 <value_of_h21>
E1 5 2 2 0 <value_of_h12>
R2 6 22 <value_of_h22>

* Feedback Network
R3 9 2 <value_of_h11f>
G2 2 0 9 4 <value_of_h21f>
E2 23 2 9 2 <value_of_h12f>
R4 4 2 <value_of_h22f>

* Output
Vout 22 0

.end