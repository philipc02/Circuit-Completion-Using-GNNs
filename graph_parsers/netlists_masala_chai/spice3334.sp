spice
* Diode Circuit
* Node List
* 1: Anode of D1
* 2: Cathode of D1 and Anode of D2
* 3: Cathode of D2

V1 5 6 DC VAB   ; Voltage source V_AB
ID 5 2 DC       ; Current source ID
D1 4 2 IS=IS N=1
D2 2 3 IS=ISR N=2

* Node connections based on reference image:
* Node 4 is connected to voltage source at node 5 (positive terminal A)
* Node 2 is common between the two diodes
* Node 6 is connected to the ground (negative terminal B)

.END